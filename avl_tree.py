import math
import shutil
from abc import abstractmethod
from collections.abc import Collection, Iterable
from typing import Any, Callable, cast, Generic, Optional, Protocol, Type, TypeVar


class ComparableTreeDataType(Protocol):
    @abstractmethod
    def __lt__(self, other: Any, /) -> bool: ...


T = TypeVar('T', bound=ComparableTreeDataType)


class AvlTreeNode(Collection, Generic[T]):
    __slots__ = 'val', 'left', 'right', 'parent', 'height', 'num_descendents'
    
    def __init__(self, init: Optional[T] = None):
        # only none for the root in an empty tree
        self.val: None | T = init
        self.left: 'None | AvlTreeNode[T]' = None
        self.right: 'None | AvlTreeNode[T]' = None
        # only None for the root
        self.parent: 'None | AvlTreeNode[T]' = None
        # height of left subtree - height of right subtree; if this subtree is balanced, should be -1, 0, or 1
        # height is max child edge count for any path; 0 if no children
        self.height: int = 0
        self.num_descendents: int = 0

    def __str__(self):
        s = str(self.val) if self.val is not None else '<Empty>'
        return f'{self.__class__.__name__}({s})'

    def __repr__(self):
        return str(self)
    
    def __len__(self):
        # the len of the tree is its number of descendents + 1 unless the value is None
        # notably, the value should only ever be None if the length is 0 and there are no descendents
        return self.num_descendents + int(self.val is not None)

    def __iter__(self):
        if self.val is not None:
            stack : 'list[AvlTreeNode[T]]' = [self]
            while stack:
                node = stack.pop()
                yield node
                node.get_children()
                stack.extend(node.get_children())
    
    def __contains__(self, x: T):
        return self.search(x)
    
    def __eq__(self, other):
        # they are equal if they have the same values and are the same length (need not have the same tree structure)
        if not isinstance(other, AvlTreeNode):
            return False
        try:
            for selfi, otheri in zip(self.sorted(), other.sorted(), strict=True):
                if selfi.val != otheri.val:
                    return False
        except ValueError:
            # they are not the same length
            return False
        return True

    def __update_parent(self, node: 'AvlTreeNode[T]'):
        """Update self.parent to replace node (one of its current children) with self."""
        p = self.parent
        if p is not None:
            if p.left is node:
                p.left = self
            elif p.right is node:
                p.right = self
            else:
                raise RuntimeError('Replaced child does not exist in parent')

    def __rotate_l(self) -> 'AvlTreeNode[T]':
        """Perform a left rotation rooted at self. Will fail if a left rotation is invalid for this node.
        
        Updates all changed heights up to the root.
        Returns the new root, replacing self from this rotation (the former right node).
        """
        # the right node becomes the new root, and the old right node's left node becomes the old root's new right child
        # the old root node becomes the left child of the new root (old right node)
        # use if the tree is right heavy and its right subtree is not left heavy
        #    *A                  C
        #   B   C      =>     *A   G
        #  D E F G            B F H I
        #       H I          D E
        # changed height: A (self), C (r)
        # this rotation is invalid otherwise, so just assume it's not None
        r = cast(AvlTreeNode[T], self.right)
        self.right = r.left
        if r.left is not None:
            r.left.parent = self
        r.left = self
        r.parent = self.parent
        self.parent = r
        r.__update_parent(self)
        self.__update_height()
        return r

    def __rotate_r(self) -> 'AvlTreeNode[T]':
        """Perform a right rotation rooted at self. Will fail if a right rotation is invalid for this node.
        
        Updates all changed heights to the root.
        Returns the new root, replacing self from this rotation (the former left node).
        """
        # the left node becomes the new root, and the old left node's right node becomes the old root's new left child
        # the old root node becomes the right child of the new root (old left node)
        # use if the tree is left heavy and its left subtree is not right heavy
        #    *A                  B
        #   B   C      =>      D  *A
        #  D E F G            H I E C
        # H I                      F G
        # changed height: A (self), B (l)
        # this rotation is invalid otherwise, so just assume it's not None
        l = cast(AvlTreeNode[T], self.left)
        self.left = l.right
        if l.right is not None:
            l.right.parent = self
        l.right = self
        l.parent = self.parent
        self.parent = l
        l.__update_parent(self)
        self.__update_height()
        return l

    def __rotate_lr(self) -> 'AvlTreeNode[T]':
        """lr rotation (AKA double left) right rotate lowest imbalanced node's right child, then left rotate lowest
        imbalanced node.

        Updates all changed heights to the root.
        Returns the new root, replacing self from this rotation.
        """
        # Use if the tree is right heavy and its right subtree is left heavy.
        cast(AvlTreeNode[T], self.right).__rotate_r()
        return self.__rotate_l()

    def __rotate_rl(self) -> 'AvlTreeNode[T]':
        """rl rotation (AKA double right) left rotate lowest imbalanced node's left child, then right rotate lowest
        imbalanced node.

        Updates all changed heights up to the root.
        Returns the new root, replacing self from this rotation.
        """
        # use if the tree is left heavy and its left subtree is right heavy
        cast(AvlTreeNode[T], self.left).__rotate_l()
        return self.__rotate_r()

    def __get_descendent_successor(self) -> 'AvlTreeNode[T] | None':
        """Get the least greater node among descendents. Return the successor or None if there are no right children."""
        # the least value in the right subtree
        child = self.right
        if child is not None:
            while child.left is not None:
                child = child.left
        return child

    def __get_descendent_predecessor(self) -> 'AvlTreeNode[T] | None':
        """Get the greatest lesser node among descendents. Return the predecessor or None if there are no left children."""
        # the greatest value in the left subtree
        child = self.left
        if child is not None:
            while child.right is not None:
                child = child.right
        return child

    def __update_height(self):
        """Update the height of the tree from self up to the root."""
        node = self
        while node is not None:
            node._update_node_height()
            node = node.parent

    def __fix_balance(self) -> tuple['AvlTreeNode[T]', bool]:
        """Fix the balance at this node if there is any balance to fix. This only checks for balance at this node. Any
        height updates that need to be performed happen automatically. Caller needs to ensure accurate height of
        children so balance can be calculated properly.
        
        Return (new_root, rotated), where new_root is the new root at this node (either the current node or the new one
        if a rotation occured). rotated is True if a rotation occured or False if one did not occur.
        """
        # check current balance to see if a rotation is necessary
        balance = self.get_balance()
        if balance < -1:
            # root is right heavy
            right_balance = self.right.get_balance() if self.right is not None else 0
            if right_balance > 0:
                # right left heavy
                return self.__rotate_lr(), True
            else:
                # right not left heavy
                return self.__rotate_l(), True
        elif balance > 1:
            # root is left heavy
            left_balance = self.left.get_balance() if self.left is not None else 0
            if left_balance < 0:
                # left right heavy
                return self.__rotate_rl(), True
            else:
                # left not right heavy
                return self.__rotate_r(), True
        # otherwise perform no rotation
        return self, False

    def __fix_tree_balance(self, rotate_once: bool) -> 'AvlTreeNode[T] | None':
        """Fix potential imbalances (if any) after an opration from self all the way up to the root. Only one rotation
        is ever performed if rotate_once is set, otherwise will keep checking until it hits the root. Heights are also
        updated on the way up whether a rotation occurs or not.

        Return new_root, where new_root is the new root if a rotation occured at the root node of this tree
        (as in the true root and not self of the caller), or None if the root has not changed.
        """
        # fix a potential imbalance (if any) caused by an operation; also update height on the way up
        # start at the descendant (self) and work up to the root; exit if an imbalance is fixed
        # after this operation, this path from the descendant up to the root is balanced and has correct height (if
        # children of descendent have the right height)
        node = self
        new_root = None
        has_rotated = False
        while node is not None:
            # update height first so the fix balance operation has the right height
            # but after one rotation, skip this since heights have been updated already by the rotation operation
            if not has_rotated:
                node._update_node_height()
            # need to check for root before rotation
            is_root = node.parent is None
            rotated_new_root, rotated = node.__fix_balance()
            if is_root:
                # technically, this should always break after this iteration in this case, but just fall through
                new_root = rotated_new_root
            if rotated:
                if rotate_once:
                    break
                has_rotated = True
                # need to traverse the tree properly from the new root position
                node = rotated_new_root
            node = node.parent
        # no rotation occured
        return new_root

    def _update_node_metadata(self, children: tuple['AvlTreeNode[T]', ...]):
        if children:
            # the height of this node is 1 + the max height of its children
            self.height = max(n.height for n in children) + 1
            # its number of descendents is the number of descendents of its children + its number of children
            self.num_descendents = sum(n.num_descendents for n in children) + len(children)
        else:
            self.height = 0
            self.num_descendents = 0

    def _update_node_height(self: 'AvlTreeNode[T]'):
        """Quickly update this node's height by looking at the heights of its children. Assumes child heights are valid.
        """
        children = self.get_children()
        self._update_node_metadata(children)

    def _calculate_height(self) -> int:
        """Returns max depth of descendents of this node as the number of child edges. If this is a leaf, the depth is
        0. If it has one level of children, its depth is 1 and so on. This calculates it manually and does not use the
        height field and should only be used for testing since it requires walking the tree.
        """
        depth = 0
        next_level = list(self.get_children())
        while next_level:
            depth += 1
            # count how many iterations it takes from a breadth first search (expanding out each level)
            # lesser values are always in the left subtree, greater values in the right subtree
            next_level = [n for node in next_level for n in node.get_children()]
        return depth

    def _calculate_balance(self) -> int:
        """Calculate the balance of this node manually, not checking the height field. This should only be used for
        testing since it requires walking the tree.
        """
        # the tree should always have a balance of -1, 0, or 1
        return (self.left._calculate_height() + 1 if self.left is not None else 0) - (self.right._calculate_height() + 1 if self.right is not None else 0)

    def _calculate_len(self) -> int:
        """Calculate the number of elements rooted at this node manually. This should only be used for testing since it
        requires walking the tree.
        """
        return sum(1 for _ in self)

    def _init_node(self):
        """Override this to provide additional node initialization. This should be what additional logic __init__ needs,
        to also be used when an empty root has a value added.
        """
        pass

    def get_balance(self) -> int:
        """Get the balance of this node based on the height of its children. A balanced node should have a balance of
        -1, 0, or 1. The convention is that balance is left height - right height. This means a positive balance is left
        heavy, and a negative balance is right heavy.
        """
        return (self.left.height + 1 if self.left is not None else 0) - (self.right.height + 1 if self.right is not None else 0)
    
    def sorted(self):
        """Return a sorted iterator over the values of the tree at this node."""
        if self.val is not None:
            stack: list['AvlTreeNode[T]'] = [self]
            done: set[T | None] = set()
            # yield nodes in order by traversing as far left as possible first, followed by thir right subtrees, then
            # traversing up the tree from there
            while stack:
                node = stack[-1]
                if node.left is None or node.left.val in done:
                    stack.pop()
                    if node.right is not None:
                        stack.append(node.right)
                    # if we want to save space for large trees, we can delete elements when they're not needed
                    done.add(node.val)
                    yield node
                else:
                    stack.append(node.left)
    
    def get_children(self) -> tuple['AvlTreeNode[T]', ...]:
        """Get a tuple of this node's children. May have 0, 1, or 2 elements. If it has 2 children, the returned order
        will always be (left, right).
        """
        return tuple(i for i in [self.left, self.right] if i is not None)
    
    def search(self, val: T) -> tuple['AvlTreeNode[T] | None', 'AvlTreeNode[T] | None']:
        """Search for value among this node and its descendents.
        
        Return (node, parent), where node is the found node (or None if no node found), and parent is the parent of the
        found node, the leaf node at the end of the search, or None if the tree is empty.
        """
        # the tree is empty, so there is no found node or parent node
        if self.val is None:
            return (None, None)
        node = self
        parent_node = node
        while node is not None:
            # the tree is invalid if any values are None after the initial check
            nodeval = cast(T, node.val)
            if val == nodeval:
                # if the node and parent_node are the same, this is the root and thus it has no parent
                return (node, parent_node if parent_node is not node else None)
            parent_node = node
            # lesser values are always in the left subtree, greater values in the right subtree
            if val < nodeval:
                node = node.left
            elif val > nodeval:
                node = node.right
        # node not found
        return (None, parent_node)

    def insert(self, to_insert_val: T) -> tuple['AvlTreeNode[T] | None', 'AvlTreeNode[T]', bool]:
        """Insert a value into the tree. If the tree is empty, insert it at the root.

        returns (new_root, node, inserted), where new_root is the new root if the root (as in the root of the whole tree
        not necessarily self) changed due to a rebalance (or None if the root did not change), node is the inserted
        node, and inserted is True if the value was inserted or False if the value was already present.
        """
        assert(to_insert_val is not None)
        # if the tree is empty, insert it at the root
        if self.val is None:
            self.val = to_insert_val
            self._init_node()
            return (None, self, True)
        node, parent_node = self.search(to_insert_val)
        if node is not None:
            # node already exists in the tree
            return (None, node, False)
        else:
            # if the parent node were None, we'd be inserting at the root; that should already be handled
            parent_node = cast(AvlTreeNode[T], parent_node)
            to_insert = self.__class__(to_insert_val)
            # the place where the node would be inserted is guaranteed to not have a child
            if to_insert_val < parent_node.val:
                parent_node.left = to_insert
            else:
                parent_node.right = to_insert
            # height is initialized to zero already
            to_insert.parent = parent_node
            # start searching for imbalance at the parent since the child will not have any imbalance
            new_root = parent_node.__fix_tree_balance(rotate_once=True)
            # the new root may be unchanged
            return (new_root, to_insert, True)

    def delete(self, val: 'T | AvlTreeNode[T]') -> tuple['AvlTreeNode[T] | None', bool]:
        """Find a node and delete it from the tree. Return the new root (possibly the same), and True if a node was
        deleted False otherwise.
        
        returns (new_root, deleted), where new_root is the new root if the root (as in the root of the whole tree not
        necessarily self) changed due to a rebalance (or None if the root did not change), and deleted is True if the
        value was deleted or False if the value was not present.
        """
        assert(val is not None)
        if isinstance(val, AvlTreeNode):
            # already found the node, don't need to find it again
            to_delete_node = val
        else:
            # try to find this value in the tree
            to_delete_node, _ = self.search(val)
            if to_delete_node is None:
                # no node was found, nothing else to do
                return (None, False)
        both_present = False
        new_root = None
        if to_delete_node.left is None:
            # right may be None, but that's fine; this means they're both None and so there are no children to move up
            new_child = to_delete_node.right
        elif to_delete_node.right is None:
            # right is None but left is not
            new_child = to_delete_node.left
        else:
            # the deleted node has two children
            both_present = True
            # this can be either predecessor or successor; it doesn't matter
            # this works because it will be guaranteed to be both greater than the left child (or is the left child)
            # and thus valid for this position, and it is guaranteed to have at most one child
            pred = to_delete_node.__get_descendent_predecessor()
            # this must be true since we have a left child
            pred = cast(AvlTreeNode[T], pred)
            pred_val = pred.val
            # recursively delete the predecessor; since it only has one child at most, will not hit this case again
            new_root, _ = self.delete(pred)
            # replace the value, but don't actually delete the node object
            to_delete_node.val = pred_val
            # in this case, there is no need to rebalance again (the recursive call does the only needed rebalance)
        if not both_present:
            # the deleted node has one child or no children
            if to_delete_node.parent is None:
                # we are deleting the root (ourselves)
                # one last thing to check: if new_child is None, that means this tree is now empty, but we can't assign to None
                if new_child is None:
                    self.val = None
                    self.height = 0
                    self.num_descendents = 0
                else:
                    # just move the one child up and that becomes the new tree
                    # in this case, its height and balance is correct so no need to rebalance
                    new_root = new_child
                    new_root.parent = None
            else:
                # not deleting the root
                if to_delete_node.parent.left is to_delete_node:
                    # deleted node is paren't left child
                    to_delete_node.parent.left = new_child
                else:
                    # deleted node is paren't right child
                    to_delete_node.parent.right = new_child
                if new_child is not None:
                    # update the moved up node's parent
                    new_child.parent = to_delete_node.parent
                # recalculate height and rebalance at the parent (any moved up child still has accurate height / balance)
                # allow for multiple rotations since deletions can cause this to happen in some cases
                new_root = to_delete_node.parent.__fix_tree_balance(rotate_once=False)
        return (new_root, True)

    def print(self, max_width=None, min_chars_per_node=3, empty_node_text='<>', node_to_str=lambda x: str(x.val)):
        """Print the tree to command line."""
        if max_width is None:
            max_width = shutil.get_terminal_size((120, 24)).columns
        MIN_WIDTH = 8
        if max_width < MIN_WIDTH:
            print(f'Can\'t print tree; available width of {max_width} needs to be at least {MIN_WIDTH}')
            return
        # sometimes there's some extra space written off the end of a line, not sure why
        max_width -= 1
        if self.val is None:
            print(str(self))
        else:
            TRUNCATE_TEXT = '..'
            ARROW_CHARS: tuple[str, str] = ('/', '\\')
            if min_chars_per_node < len(TRUNCATE_TEXT):
                print(f'Can\'t print tree; min chars per node {min_chars_per_node} needs to be at least {len(TRUNCATE_TEXT)}')
                return
            # in order from left to right; each level has 2^n (where the first is n=0) entries
            next_level : list[AvlTreeNode[T] | None] = [self]
            current_level = 0
            while any(next_level):
                node_count = len(next_level)
                total_spaces_between_nodes = node_count - 1
                space_per_node = ((max_width - total_spaces_between_nodes) / node_count)
                temp = math.floor(space_per_node)
                unused_space_per_node = space_per_node - temp
                space_per_node = temp
                arrow_left_align_space = (space_per_node - len(ARROW_CHARS[0])) // 2
                arrow_right_align_space = (space_per_node - len(ARROW_CHARS[0])) - arrow_left_align_space
                # end early and indicate that there's more below; there are too many nodes to effectively print any further down
                if space_per_node < min_chars_per_node:
                    print(f'{"...": ^{max_width}}')
                    break
                this_level = next_level
                next_level = []
                to_print = []
                accumulated_space = 0.0
                for idx, node in enumerate(this_level):
                    # don't add the extra space for the last node
                    space_after = '' if idx == len(this_level) - 1 else ' '
                    if accumulated_space >= 1.0:
                        extra_space = 1
                        accumulated_space -= 1.0
                    else:
                        extra_space = 0
                    if node is None:
                        node_text = empty_node_text
                        next_level.extend([None, None])
                    else:
                        node_text = node_to_str(node)
                        next_level.extend((node.left, node.right))
                    # truncate node text that is too long
                    if len(node_text) > space_per_node:
                        node_text = node_text[:space_per_node - len(TRUNCATE_TEXT)] + TRUNCATE_TEXT
                    # for all but the first line, print arrows pointing to the nodes on the next line first
                    if current_level != 0:
                        # alternate printing left and right arrows (0=left, 1=right)
                        side = idx % 2
                        arrow_text = ARROW_CHARS[side] if node is not None else ''
                        if node is not None:
                            left_align_char = ' ' if side == 0 else '-'
                            right_align_char = '-' if side == 0 else ' '
                            arrow_text = left_align_char * (arrow_left_align_space + extra_space) + arrow_text + right_align_char * arrow_right_align_space + space_after
                        else:
                            # blank entry with just spaces to align later arrows
                            # add 1 because of the space between nodes
                            arrow_text = f' ' * (space_per_node + extra_space) + space_after
                        print(arrow_text, end='')
                    to_print.append(f'{node_text: ^{space_per_node + extra_space}}{space_after}')
                    accumulated_space += unused_space_per_node
                # now end the line of arrows and print the next line of nodes
                print('')
                print(''.join(to_print))
                current_level += 1


class GenericAvlTree(Collection, Generic[T]):
    """Generic avl tree; can construct with a node type or override methods as desired to specify a node type."""
    __slots__ = ('_root')

    def __init__(self, tree_node_cls: Type[AvlTreeNode], init: Optional[Iterable[T]] = None):
        """Initialize the tree (with a node type), optionally with an iterable of values to initially insert."""
        self._root: AvlTreeNode[T]
        if init:
            first, *rest = init
            self._root = tree_node_cls(first)
            for val in rest:
                self.insert(val)
        else:
            self._root = tree_node_cls()

    def __len__(self):
        """The length of a tree is the number of values, and it is stored in the tree, so calculating it is a constant
        time operation.
        """
        return len(self._root)

    def __iter__(self):
        for node in self._root:
            yield node.val
    
    def __str__(self):
        return f'{self.__class__.__name__}({str(list(self))})'
    
    def __repr__(self):
        return str(self)
    
    def __contains__(self, x: T):
        return x in self._root

    def __eq__(self, other):
        """Trees are equal if they have the same contents."""
        if not isinstance(other, GenericAvlTree):
            return False
        return self._root == other._root
    
    def clear(self):
        """Removes all elements from the tree."""
        self._root = self._root.__class__()

    def sorted(self) -> Iterable[T]:
        """Return a sorted iterator over the values in the tree."""
        for node in self._root.sorted():
            if node.val is not None:
                yield node.val

    def insert(self, val: T):
        """Insert a value into the tree. Return True if the value was inserted, False if it was already present."""
        node, _, inserted = self._root.insert(val)
        if node is not None:
            self._root = node
        return inserted

    def delete(self, val: T):
        """Insert a value from the tree. Return True if the value was deleted, False if it was not present."""
        node, deleted = self._root.delete(val)
        if node is not None:
            self._root = node
        return deleted

    def search(self, val: T):
        """Return True if a value is in the tree. False if not."""
        return self._root.search(val)[0] is not None

    def extend(self, vals: Iterable[T]) -> int:
        """Add an iterable of elements to the tree. Returns the number of elements inserted."""
        inserted = 0
        for val in vals:
            inserted += int(self.insert(val))
        return inserted

    def print(self, *args, **kwargs):
        """Try to print the tree to console in a human-readable format as space allows. Printing stops when the entire
        tree has been printed, or there is no longer enough space available to reasonably print a line.

        max_width is the amount of space available to print each line; None (the default) uses the current console
        width.

        min_chars_per_node is the minimum amount of space to have in a line for each node. When this can no longer be
        satisfied, stop printing.

        empty_node_text is what to print as a placeholder for a node that doesn't exist (i.e. it's parent doesn't have a
        child in this slot).
        """
        self._root.print(*args, **kwargs)

    @staticmethod
    def test(tree_node_cls: Type[AvlTreeNode], iters=1, iters_per_iter=1000, delete_prob=.1, print_time=True, print_tree=False):
        """Run tests. Will throw an AssertionError if there is an error."""
        import random
        import time
        start_time = time.time()
        for _ in range(iters):
            vals: set[int] = set()
            tree: GenericAvlTree[int] = GenericAvlTree(tree_node_cls)
            # the tree should start out empty
            assert(len(tree) == 0)
            assert(tree._root.val is None)
            # insert and delete a group of nodes, adding them both to the tree and to a set
            for _ in range(iters_per_iter):
                delete = random.random() <= delete_prob
                if delete:
                    if len(vals) > 0:
                        # making a random choice from a set is a O(N) operation, so this is inefficient
                        # but for a test, it's fine
                        val = random.choice(tuple(vals))
                        assert(tree.delete(val))
                        vals.remove(val)
                else:
                    val = random.randint(-100000, 100000)
                    already_exists = val in vals
                    assert(tree.insert(val) != already_exists)
                    vals.add(val)
            # they should now have the same number of elements and when sorted should be the same
            assert(len(tree) == len(vals))
            assert(len(tree) == tree._root._calculate_len())
            # test sorting
            assert(list(tree.sorted()) == sorted(vals))
            none_parent_count = 0
            seen_vals: set[int] = set()
            for el in tree._root:
                # the balance should be -1, 0, or 1, the calculated balance and balance from height should match
                # the parent should also have that node as one of its children
                balance = el.get_balance()
                assert(balance == el._calculate_balance())
                assert(abs(balance) <= 1)
                if el.parent is None:
                    none_parent_count += 1
                else:
                    assert(el.parent.left is el or el.parent.right is el)
                # if the tree is not empty, then no value should be None and there should be no duplicates
                assert(el.val is not None)
                assert(el.val not in seen_vals)
                # the length of each node should be correct
                assert(len(el) == el._calculate_len())
                # the height of each node should be correct
                assert(el.height == el._calculate_height())
                seen_vals.add(el.val)
            # there should exactly one element with no parent (the root)
            # also have to check manually in the empty case since there is nothing to iterate over
            assert(none_parent_count == 1 or (len(tree) == 0 and tree._root.parent is None))
            assert(bool(tree) == True or len(tree) == 0)
            if print_tree:
                tree.print()
            for val in vals:
                # the value should not be inserted (since it already exists)
                # the value should be found, and be able to be deleted
                assert(not tree.insert(val))
                assert(val in tree)
                assert(tree.search(val))
                assert(tree.delete(val))
            # after deleting everything, the tree should be empty
            assert(len(tree) == 0)
            assert(tree._root._calculate_len() == 0)
            assert(tree._root.val is None)
            assert(list(tree) == [])
            assert(bool(tree) == False)
        end_time = time.time()
        total_time = end_time - start_time
        if print_time:
            print(f'Test successful with {iters} iterations and {iters_per_iter} steps per iteration')
            print(f'Total time of {total_time:.2f}s and average time of {(total_time / iters):.2f}s per iteration')


class AvlTree(GenericAvlTree):
    """Avl tree using AvlTreeNode."""

    def __init__(self, *args, **kwargs):
        super().__init__(AvlTreeNode, *args, **kwargs)

    @staticmethod
    def test(*args, **kwargs):
        super(AvlTree, AvlTree).test(AvlTreeNode, *args, **kwargs)


if __name__ == '__main__':
    AvlTree.test()
