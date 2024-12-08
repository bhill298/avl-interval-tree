import math
import shutil
from abc import abstractmethod
from collections.abc import Collection, Iterable
from typing import Any, cast, Generic, Optional, TypeVar, Protocol

# https://stackoverflow.com/questions/17466218/what-are-the-differences-between-segment-trees-interval-trees-binary-indexed-t
# A segment tree stores intervals and is optimized for "which of these intervals contains a given point" queries.
# An interval tree stores intervals as well, but optimized for "which of these intervals overlap with a given interval" queries. It can also be used for point queries - similar to segment tree.

STRIDE = 0x1000
# [start, end), value
data = [(STRIDE * 0, STRIDE * 5, 0x0), (STRIDE * 3, STRIDE * 12, 0x5), (STRIDE * 5, STRIDE * 6, 0x1), (STRIDE * 100, STRIDE * 102, 0x2), (STRIDE * 20, STRIDE * 22, 0x3), (STRIDE * 1000, STRIDE * 1010, 0x4)]
# contains 0x0
TEST1 = STRIDE * 0
# contains 0x0, 0x5
TEST2 = STRIDE * 4
# contains 0x5, 0x1
TEST3 = STRIDE * 5
# contains 0x2
TEST4 = (STRIDE * 102) - 1
# contains 0x3
TEST5 = STRIDE * 21
# contains nothing
TEST6 = (STRIDE * 1000) - 1

class ComparableTreeDataType(Protocol):
    @abstractmethod
    def __lt__(self, other: Any, /) -> bool: ...

T = TypeVar('T', bound=ComparableTreeDataType)


class AvlTreeNode(Collection, Generic[T]):
    __slots__ = 'val', 'left', 'right', 'parent', 'height'
    
    def __init__(self, init: Optional[Iterable[T]] = None):
        self.val: None | T = None
        self.left: 'None | AvlTreeNode[T]' = None
        self.right: 'None | AvlTreeNode[T]' = None
        # only None for the root
        self.parent: 'None | AvlTreeNode[T]' = None
        # height of left subtree - height of right subtree; if this subtree is balanced, should be -1, 0, or 1
        # height is max child edge count for any path; 0 if no children
        self.height: int = 0
        if init is not None:
            for val in init:
                self.insert(val)

    def __str__(self):
        s = str(self.val) if self.val is not None else '<Empty>'
        return f'AvlTreeNode({s})'

    def __repr__(self):
        return str(self)
    
    def __len__(self):
        # return the number of descendents
        return sum(1 for _ in self)

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
        try:
            for selfi, otheri in zip(self.sorted(), other.sorted(), strict=True):
                if selfi.val != otheri.val:
                    return False
        except ValueError:
            # they are not the same length
            return False
        return True

    def __get_val(self) -> T:
        # the only time val should be none is in an empty root; outside of those cases, can use this
        return cast(T, self.val)
    
    def __update_parent(self, node: 'AvlTreeNode[T]'):
        # update self.parent to now link to itself, replacing node
        if self.parent is not None:
            if self.parent.left is node:
                self.parent.left = self
            elif self.parent.right is node:
                self.parent.right = self
            else:
                raise RuntimeError('Replaced child does not exist in parent')

    def __rotate_l(self) -> 'AvlTreeNode[T]':
        # the right node becomes the new root, and the old right node's left node becomes the old root's new right child
        # the old root node becomes the left child of the new root (old right node)
        # use if the tree is right heavy and its right subtree is not left heavy
        #    *A                  C
        #   B   C      =>     *A   G
        #  D E F G            B F H I
        #       H I          D E
        # changed height: A (self), C (r)
        assert(self.right is not None and self.right.left is not None)
        r = self.right
        self.right = r.left
        if r.left is not None:
            r.left.parent = self
        r.left = self
        r.parent = self.parent
        self.parent = r
        r.__update_parent(self)
        r.__update_height(self)
        return r

    def __rotate_r(self) -> 'AvlTreeNode[T]':
        # the left node becomes the new root, and the old left node's right node becomes the old root's new left child
        # the old root node becomes the right child of the new root (old left node)
        # use if the tree is left heavy and its left subtree is not right heavy
        #    *A                  B
        #   B   C      =>      D  *A
        #  D E F G            H I E C
        # H I                      F G
        # changed height: A (self), B (l)
        assert(self.left is not None and self.left.right is not None)
        l = self.left
        self.left = l.right
        if l.right is not None:
            l.right.parent = self
        l.right = self
        l.parent = self.parent
        self.parent = l
        l.__update_parent(self)
        l.__update_height(self)
        return l

    def __rotate_lr(self) -> 'AvlTreeNode[T]':
        # (AKA double left) right rotate lowest imbalanced node's right child, then left rotate lowest imbalanced node
        # use if the tree is right heavy and its right subtree is left heavy
        assert(self.right is not None)
        self.right.__rotate_r()
        return self.__rotate_l()

    def __rotate_rl(self) -> 'AvlTreeNode[T]':
        # (AKA double right) left rotate lowest imbalanced node's left child, then right rotate lowest imbalanced node
        # use if the tree is left heavy and its left subtree is right heavy
        assert(self.left is not None)
        self.left.__rotate_l()
        return self.__rotate_r()

    def __calculate_depth(self) -> int:
        """Returns max depth of descendents of this node as the number of child edges. If this is a leaf, the
        depth is 0. If it has one level of children, its depth is 1 and so on."""
        depth = 0
        stack = [n for n in self.get_children()]
        while stack:
            depth += 1
            stack.extend(n for node in stack.pop() for n in node.get_children())
        return depth

    def __get_successor(self) -> 'AvlTreeNode[T] | None':
        """Get the least greater node among descendents. Return succ. succ is None if this is the greatest value."""
        # the least value in the right subtree
        child = self.right
        if child is not None:
            while child.left is not None:
                child = child.left
        return child

    def __get_predecessor(self) -> 'AvlTreeNode[T] | None':
        """Get the greatest lesser node among descendents. Return pred. pred is None if this is the least value."""
        # the greatest value in the left subtree
        child = self.left
        if child is not None:
            while child.right is not None:
                child = child.right
        return child

    def __update_node_height(self: 'AvlTreeNode[T]'):
        children = self.get_children()
        if children:
            self.height = max(n.height for n in children) + 1
        else:
            self.height = 0

    def __update_height(self, affected_node: 'AvlTreeNode[T]'):
        # rebalance along the path from the affected node to the root (self)
        # don't do any checks here, since we assume this is getting called with a valid tree
        node = affected_node
        while node is not self:
            node.__update_node_height()
            # assume node is not None; if this is ever None, the tree is invalid
            node = cast(AvlTreeNode[T], node.parent)

    def __calculate_balance(self) -> int:
        # the tree should always have a balance of -1, 0, or 1
        return (self.left.__calculate_depth() + 1 if self.left is not None else 0) - (self.right.__calculate_depth() + 1 if self.right is not None else 0)
    
    def __fix_balance(self) -> tuple['AvlTreeNode[T]', bool]:
        # perform balance update based on current balance; return new root if it changes or old root
        # this only performs rotations on the root and won't recursively check children
        # height values are also assumed to be accurate
        # any height updates that need to be performed are done automatically
        balance = self.get_balance()
        if balance < -1:
            # root is right heavy
            right_balance = self.right.get_balance() if self.right is not None else 0
            if right_balance > 1:
                # right left heavy
                return self.__rotate_lr(), True
            else:
                # right not left heavy
                return self.__rotate_l(), True
        elif balance > 1:
            # root is left heavy
            left_balance = self.left.get_balance() if self.left is not None else 0
            if left_balance < -1:
                # left right heavy
                return self.__rotate_rl(), True
            else:
                # left not right heavy
                return self.__rotate_r(), True
        # otherwise perform no rotation
        return self, False

    def __fix_tree_balance(self, descendant: 'AvlTreeNode[T]') -> tuple['AvlTreeNode[T]', bool]:
        # fix a potential imbalance (if any) caused by an operation; also update height on the way up
        # start at the descendant and work up to the root; exit if an imbalance is fixed
        # after this operation, this path from the descendant up to the root is balanced and has correct height (if
        # children of descendent have the right height)
        node = descendant
        while True:
            # update height first so the fix balance operation has the right height
            node.__update_node_height()
            new_root, rotated = node.__fix_balance()
            if rotated:
                # TODO: I believe that if an imbalance occurs from a single operation, fixing it at the lowest
                #  point should fix the tree imbalance in all cases; is this true? (once confirmed remove this assert)
                assert(self.__fix_tree_balance(descendant)[1] == False)
                if node is self:
                    # the root node has been replaced with a new root via rotation; return the new root
                    return new_root, True
                else:
                    # a rotation occured, but it was not at the root so it is unchanged
                    return self, True
            elif node is self:
                # done
                break
            # this is an invalid call if the parent is ever None; should never happen
            node = cast('AvlTreeNode[T]', node.parent)
        # no rotation occured
        return self, False

    def get_balance(self) -> int:
        # the convention is that balance is left height - right height
        # this means that a positive balance is "left heavy" and a negative balance is "right heavy"
        # -1, 0, or 1 indicates a sufficiently balanced node
        return (self.left.height + 1 if self.left is not None else 0) - (self.right.height + 1 if self.right is not None else 0)
    
    def sorted(self):
        if self.val is not None:
            stack: list['AvlTreeNode[T]'] = [self]
            done: set['AvlTreeNode[T]'] = set()
            while stack:
                node = stack[-1]
                if node.left is None or node.left in done:
                    stack.pop()
                    if node.right is not None:
                        stack.append(node.right)
                    # if we want to save space for large trees, we can delete elements when they're not needed
                    done.add(node)
                    yield node
                else:
                    stack.append(node.left)
    
    def get_children(self) -> tuple['AvlTreeNode[T]', ...]:
        return tuple(i for i in [self.left, self.right] if i is not None)
    
    def search(self, val: T) -> tuple['AvlTreeNode[T] | None', 'AvlTreeNode[T] | None']:
        """Search for value. Return the found node (or None), and its parent / leaf node encountered at end of search if not found (or None if tree is empty)."""
        # the tree is empty, so there is no found node or parent node
        if self.val is None:
            return (None, None)
        node = self
        parent_node = node
        while node is not None:
            nodeval = node.__get_val()
            if val == nodeval:
                # if the node and parent_node are the same, this is the root and thus it has no parent
                return (node, parent_node if parent_node is not node else None)
            parent_node = node
            if val < nodeval:
                node = node.left
            elif val > nodeval:
                node = node.right
        # node not found
        return (None, parent_node)

    def insert(self, to_insert_val: T) -> tuple['AvlTreeNode[T]', bool]:
        # if the tree is empty, insert it at the root
        if self.val is None:
            self.val = to_insert_val
            return (self, True)
        node, parent_node = self.search(to_insert_val)
        if node is not None:
            # node already exists in the tree
            return (self, False)
        else:
            # if the parent node were None, we'd be inserting at the root; that should already be handled
            assert(parent_node is not None)
            to_insert = AvlTreeNode([to_insert_val])
            # the place where the node would be inserted is guaranteed to not have a child
            if to_insert_val < parent_node.val:
                parent_node.left = to_insert
            else:
                parent_node.right = to_insert
            # height is initialized to zero already
            to_insert.parent = parent_node
            # start searching for imbalance at the parent since the child will not have any imbalance
            new_root, _ = self.__fix_tree_balance(parent_node)
            # the new root may be unchanged
            return (new_root, True)

    def delete(self, val: 'T | AvlTreeNode[T]') -> tuple['AvlTreeNode[T]', bool]:
        """Find a node and delete it from the tree. Return the new root (possibly the same), and True if a node was deleted False otherwise."""
        if isinstance(val, AvlTreeNode):
            # already found the node, don't need to find it again
            node = val
        else:
            # try to find this value in the tree
            node, _ = self.search(val)
        if node is None:
            # no node was found, nothing else to do
            return (self, False)
        # TODO: rebalance and recompute height if needed
        both_present = False
        if node.left is None:
            # right may be None, but that's fine; this means they're both None and so there are no children to move up
            new_child = node.right
        elif node.right is None:
            # right is None but left is not
            new_child = node.left
        else:
            # both are not None
            both_present = True
            # this can be either predecessor or successor, it doesn't matter
            # this works because it will be guaranteed to be both greater than the left child (or is the left child)
            # and thus valid for this position, and it is guaranteed to have at most one child
            pred = node.__get_predecessor()
            # this must be true since we have a left child
            assert(pred is not None)
            pred_val = pred.val
            # recursively delete this; since it only has one child at most, will not hit this case again
            self, _ = self.delete(pred)
            # replace the value, but don't actually finish removing the node
            node.val = pred_val
            # TODO: in this case, there is no need to rebalance again (the recursive call does the only needed rebalance)
        if not both_present:
            if node.parent is None:
                # we are deleting the root (ourselves)
                # one last thing to check: if new_child is None, that means this tree is now empty, but we can't assign to None
                if new_child is None:
                    self.val = None
                else:
                    self = new_child
            elif node.parent.left is node:
                # deleting left child
                node.parent.left = new_child
            else:
                # deleting right child
                node.parent.right = new_child
            if new_child is not None and node.parent is not None:
                new_child.parent = node.parent
        return (self, True)

    def print(self, max_width=None, min_chars_per_node=3, empty_node_text='<>'):
        if max_width is None:
            max_width = shutil.get_terminal_size((120, 24)).columns
        # sometimes there's some extra space written off the end of a line, not sure why
        max_width -= 1
        assert(max_width >= 7)
        if self.val is None:
            print(str(self))
        else:
            TRUNCATE_TEXT = '..'
            ARROW_CHARS: tuple[str, str] = ('/', '\\')
            assert(min_chars_per_node >= len(TRUNCATE_TEXT))
            # in order from left to right; each level has 2^n (where the first is 0) entries
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
                    # don't an the extra space for the last node
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
                        node_text = str(node.val)
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


# segment tree https://en.wikipedia.org/wiki/Segment_tree
# http://www.cs.emory.edu/~cheung/Courses/253/Syllabus/Trees/
# https://www.cise.ufl.edu/~nemo/cop3530/AVL-Tree-Rotations.pdf
class AvlTree(Collection, Generic[T]):
    __slots__ = ('root')

    def __init__(self, init: Optional[Iterable[T]] = None):
        self.root = AvlTreeNode(init)

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        for node in self.root:
            yield node.val
    
    def __str__(self):
        return f'AvlTree({str(list(self))})'
    
    def __repr__(self):
        return str(self)
    
    def __contains__(self, x: T):
        return x in self.root

    def __eq__(self, other):
        return self.root == other.root
    
    def sorted(self) -> Iterable[T]:
        for node in self.root.sorted():
            if node.val is not None:
                yield node.val
    
    def get_balance(self):
        return self.root.get_balance()
    
    def is_empty(self) -> bool:
        root = self.root
        return root.val is None and root.left is None and root.right is None

    def insert(self, val: T):
        node, inserted = self.root.insert(val)
        self.root = node
        return inserted

    def delete(self, val: T):
        node, deleted = self.root.delete(val)
        self.root = node
        return deleted

    def search(self, val: T):
        return self.root.search(val)[0] is not None

    def print(self, *args, **kwargs):
        self.root.print(*args, **kwargs)

    @staticmethod
    def test(iters=1, iters_per_iter=1000, delete_prob=.1, print_time=True, print_tree=False):
        import random
        import time
        start_time = time.time()
        for _ in range(iters):
            vals: set[int] = set()
            tree: AvlTree[int] = AvlTree()
            # the tree should start out empty
            assert(len(tree) == 0)
            assert(tree.root.val is None)
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
            assert(list(tree.sorted()) == sorted(vals))
            none_parent_count = 0
            seen_vals: set[int] = set()
            for el in tree.root:
                # the balance should be -1, 0, or 1, the calculated balance and balance from height should match
                # the parent should also have that node as one of its children
                balance = el.get_balance()
                assert(balance == el.__calculate_balance())
                assert(abs(balance) <= 1)
                if el.parent is None:
                    none_parent_count += 1
                else:
                    assert(el.parent.left is el or el.parent.left is el)
                # if the tree is not empty, then no value should be None and there should be no duplicates
                assert(el.val is not None)
                assert(el.val not in seen_vals)
                seen_vals.add(el.val)
            # there should exactly one element with no parent (the root)
            assert(none_parent_count == 1)
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
            assert(tree.root.val is None)
            assert(list(tree) == [])
        end_time = time.time()
        if print_time:
            print(f'Test successful with {iters} iterations and {iters_per_iter} steps per iteration')
            print(f'Average time of {((end_time - start_time) / iters):.2f}s per iteration')


if __name__ == '__main__':
    AvlTree.test()
