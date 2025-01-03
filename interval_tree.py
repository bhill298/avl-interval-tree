import operator
import time
from abc import abstractmethod
from collections.abc import Iterable
from itertools import chain
from typing import Any, cast, Generic, Optional, Protocol, TypeVar

from avl_tree import AvlTreeNode, GenericAvlTree


class IntervalDataType(Protocol):
    @abstractmethod
    def __lt__(self, other: Any, /) -> bool: ...
    @abstractmethod
    def __le__(self, other: Any, /) -> bool: ...


T = TypeVar('T', bound=IntervalDataType)
_opsgt = [operator.gt, operator.ge]
_opslt = [operator.lt, operator.le]


class IntervalTreeNode(AvlTreeNode, Generic[T]):
    __slots__ = 'max_upper_value', 'data'

    def _init_node(self):
        super()._init_node()
        # The value initially contains both the interval and the data in a single tuple. This needs to be unpacked.
        dat = cast(tuple[T, T, Any], self.val)
        try:
            self.max_upper_value = dat[1]
            self.val = dat[:2]
            self.data = dat[2]
        except TypeError:
            # self.val should only ever be None for initial root creation, so handle this case in an exception
            self.max_upper_value = None
            self.data: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val: None | tuple[T, T]

    def __contains__(self, val) -> bool:
        """Returns True if the point or interval overlaps."""
        return any(True for _ in self.interval_search(*val))

    def __eq__(self, other) -> bool:
        # they are equal if they have the same values and are the same length (need not have the same tree structure)
        if not isinstance(other, IntervalTreeNode):
            return False
        try:
            for selfi, otheri in zip(self.sorted(), other.sorted(), strict=True):
                if selfi.val != otheri.val or cast(IntervalTreeNode, selfi).data != cast(IntervalTreeNode, otheri).data:
                    return False
        except ValueError:
            # they are not the same length
            return False
        return True

    @staticmethod
    def __overlapswith(i1: tuple[T, T], i2: tuple[T, T]) -> bool:
        """Check if two points or intervals overlap. For points, specify the same start and end point."""
        # branchless implementation
        # if i1 is a point, use >=
        opgt = _opsgt[int(i1[0] == i1[1])]
        # if i2 is a point, use <=
        oplt = _opslt[int(i2[0] == i2[1])]
        # check i1 end is >(=) i2 start and i1 start is <(=) i2 end
        # if i1 is a point, its start and end are the same, and so the end can be treated as inclusive, use >=
        # for i2, the same but use <=
        return opgt(i1[1], i2[0]) and oplt(i1[0], i2[1])

    def _update_node_metadata(self, children):
        super()._update_node_metadata(children)
        node = cast(IntervalTreeNode, self)
        children = cast(tuple[IntervalTreeNode, ...], children)
        # this should never fire with an empty root (the only time max_upper_value will ever be None)
        # the new max upper value is the max of child upper values and our own max interval
        # note that we recheck our max interval and not our current max upper value in case of rotations or reinsertion
        node.max_upper_value = max(chain((cast(T, i.max_upper_value) for i in children), [cast(tuple, node.val)[1]]))

    def insert(self, to_insert_val: tuple[T, T, Any], *args, **kwargs):
        """Insert an interval and a data value associated with the interval in the form (min, max, data). A point can be
        inserted by making min and max the same.
        """
        assert(to_insert_val[1] >= to_insert_val[0])
        new_root, node, inserted = super().insert(to_insert_val, *args, **kwargs)
        node = cast(IntervalTreeNode, node)
        # this will update the node value if it already existed
        node.data = to_insert_val[2]
        return new_root, node, inserted

    def interval_search(self, min: T, max: Optional[T] = None, exact: bool = False) -> 'Iterable[IntervalTreeNode[T]]':
        """Returns an iterator over the matching nodes overlapping the range [min, max). Pass in a single value as min
        to search for all intervals that a point overlaps with. exact determines if an interval has to match exactly
        (and at most one interval will be returned in that case).
        """
        # tree is empty
        if self.val is None:
            return
        # point interval
        if max is None:
            max = min
        to_search: list[Any] = [self]
        while to_search:
            node = to_search.pop()
            # this needs to be strictly greater than in case one of the intervals is a point interval, in which case its
            # max and min are equal and thus it could still be in range
            if min > node.max_upper_value:
                # the start of this interval is greater than the end of this node and all children
                continue
            if exact:
                if node.val == (min, max):
                    yield node
                    # there can only be exactly one match
                    return
            else:
                if self.__overlapswith((min, max), node.val):
                    # intervals / points overlap
                    yield node
            if node.left is not None:
                # search left children (check unconditionally; know that the max may be greater than our min, so nodes
                # in the left subtree may overlap us now matter how small their start is)
                to_search.append(node.left)
            if max < node.val[0]:
                # the end of the interval being searched for is less than the node's start (all nodes in right subtree
                # start to the right of this interval's bounds)
                continue
            if node.right is not None:
                # search right children (right nodes may have a start within this interval, so they may overlap)
                to_search.append(node.right)
        

class IntervalTree(GenericAvlTree, Generic[T]):
    """Interval tree based on a self-balancing AVL tree. Intervals are stored in the form [max, min) and are sorted
    based on interval start (with the interval end as a tiebreaker). Each interval may store a piece of data that can be
    retrieved when searching the tree. Additionally, the max upper value of each node and all descendents is stored to
    make searches faster. Point intervals and queries, where the min and max are the same, are allowed.
    """
    def __init__(self, *args, **kwargs):
        self._root: IntervalTreeNode
        super().__init__(IntervalTreeNode, *args, **kwargs)

    def sorted(self) -> Iterable[tuple[T, T]]:
        return super().sorted()

    def insert(self, min: T, max: T, data: Any):
        return super().insert((min, max, data))

    def delete(self, min: T, max: T):
        return super().delete((min, max))

    def search(self, min: T, max: Optional[T] = None, exact: bool = False) -> Iterable[tuple[tuple[T, T], Any]]:
        """Returns an iterator over the matching ranges and values overlapping the range [min, max). Pass in a single
        value as min to search for all intervals that a point overlaps with. exact determines if an interval has to
        match exactly (and at most one interval will be returned in that case).

        The iterator produces values in the form ((min, max), data).
        """
        for el in self._root.interval_search(min, max, exact):
            yield (cast(tuple[T, T], el.val), el.data)

    def extend(self, vals: Iterable[tuple[T, T, Any]]) -> int:
        return super().extend(vals)

    @staticmethod
    def test(print_time=True, print_ranges=False, print_tree=False):
        STRIDE = 0x1000
        # [start, end), value
        data = [(STRIDE * 0, STRIDE * 5, 0x0),
                (STRIDE * 3, STRIDE * 12, 0x5),
                (STRIDE * 3, STRIDE * 4, 0x20),
                (STRIDE * 5, STRIDE * 6, 0x1),
                (STRIDE * 100, STRIDE * 102, 0x2),
                (STRIDE * 20, STRIDE * 22, 0x3),
                (STRIDE * 1000, STRIDE * 1010, 0x4),
                # point range
                (STRIDE * 1005, STRIDE * 1005, 0x10)]
        tests = [
            # contains 0x0
            (STRIDE * 0, (0x0,)),
            # contains 0x0, 0x5, 0x20
            (int(STRIDE * 3.5), (0x0, 0x5, 0x20)),
            # same as above but in range qeury form
            ((int(STRIDE * 3.5), int(STRIDE * 3.5)), (0x0, 0x5, 0x20)),
            # contains 0x0, 0x5
            (STRIDE * 4, (0x0, 0x5)),
            # contains 0x5, 0x1
            (STRIDE * 5, (0x5, 0x1)),
            # contains 0x2
            ((STRIDE * 102) - 1, (0x2,)),
            # contains 0x1
            (STRIDE * 20, (0x3,)),
            # contains 0x3
            (STRIDE * 21, (0x3,)),
            # range query
            ((STRIDE * 0, STRIDE * 20), (0x0, 0x5, 0x20, 0x1)),
            # range query
            ((STRIDE * 20, (STRIDE * 100) + 1), (0x3, 0x2)),
            # in point and range
            (STRIDE * 1005, (0x4, 0x10)),
            ((STRIDE * 1004, (STRIDE * 1005) + 1), (0x4, 0x10)),
            ((STRIDE * 1004, STRIDE * 1005), (0x4,)),
            # contains nothing
            ((STRIDE * 1000) - 1, None),
            ((STRIDE * 1010), None),
            ((STRIDE * 1010, STRIDE * 1020), None),
        ]
        start_time = time.time()
        tree = IntervalTree()
        tree2 = IntervalTree()
        added = 0
        for d in data:
            added += int(tree.insert(d[0], d[1], d[2]))
            assert((d[0], d[1]) in tree)
            if print_tree:
                tree.print(node_to_str=lambda x: f'[{hex(x.val[0])}, {hex(x.val[1])}); {hex(x.max_upper_value)}; {x.data}')
            assert(len(tree) == added)
        assert(tree != tree2)
        for d in reversed(data):
            tree2.insert(d[0], d[1], d[2])
        # even with different insertion order, the trees should be equal
        assert(tree == tree2)
        for d in data:
            tree2.insert(d[0], d[1], 'some data')
        # one data value has changed, so they should be unequal
        assert(tree != tree2)
        for test in tests:
            val = None
            if isinstance(test[0], tuple):
                # range query
                vals = list(tree.search(test[0][0], test[0][1]))
                if print_ranges:
                    print(f'[{hex(test[0][0])}, {hex(test[0][1])})')
            else:
                vals = list(tree.search(test[0]))
                if print_ranges:
                    print(hex(test[0]))
            if print_ranges:
                print([f'[{hex(i[0][0])}, {hex(i[0][1])}); {i[1]}' for i in vals])
            if vals:
                val = [v[1] for v in vals]
            else:
                val = None
            if val is None:
                assert(test[1] is None)
            else:
                # strict asserts they are the same length
                for i1, i2 in zip(sorted(val), sorted(test[1]), strict=True):
                    assert(i1 == i2)
        # now delete everything
        deleted = 0
        for d in data:
            tree.delete(d[0], d[1])
            deleted += 1
            if print_tree:
                tree.print(node_to_str=lambda x: f'[{hex(x.val[0])}, {hex(x.val[1])}); {hex(x.max_upper_value)}; {x.data}')
            assert(len(tree) == len(data) - deleted)
        total_time = time.time() - start_time
        if print_time:
            print(f'Tests passed with total time of {total_time:.2f}s')


if __name__ == '__main__':
    IntervalTree.test()
