import operator
import time
from abc import abstractmethod
from collections.abc import Iterable
from itertools import chain
from typing import Any, cast, Generic, Optional, Protocol, TypeVar

from avl_tree import AvlTreeNode, GenericAvlTree

# https://en.wikipedia.org/wiki/Interval_tree#Augmented_tree


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
        self._init_node()

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
        assert(to_insert_val[1] >= to_insert_val[0])
        new_root, node, inserted = super().insert(to_insert_val, *args, **kwargs)
        node = cast(IntervalTreeNode, node)
        # this will update the node if it already existed
        node.data = to_insert_val[2]
        return new_root, node, inserted

    def interval_search(self, min: T, max: T) -> 'Iterable[IntervalTreeNode[T]]':
        # tree is empty
        if self.val is None:
            return
        to_search: list[Any] = [self]
        while to_search:
            node = to_search.pop()
            node_max_upper_value = node.max_upper_value
            # this needs to check if in range first before looking at the max upper value
            # there is an edge case where this node is a point interval where its max is equal to its point, in which
            # case it could be a valid overlap that would be skipped if the min if that same value
            if self.__overlapswith((min, max), node.val):
                # intervals / points overlap
                yield node
            if min >= node_max_upper_value:
                # the start of this interval is greater than the end of this node and all children
                continue
            if node.left is not None:
                # search left children
                to_search.append(node.left)
            if max < node.val[0]:
                # the end of this interval is less than the start of this node
                continue
            if node.right is not None:
                # search right children
                to_search.append(node.right)
        

class IntervalTree(GenericAvlTree, Generic[T]):
    def __init__(self, *args, **kwargs):
        self._root: IntervalTreeNode
        super().__init__(IntervalTreeNode, *args, **kwargs)

    def __contains__(self, min: T, max: T):
        return (min, max) in self._root

    def sorted(self) -> Iterable[tuple[T, T]]:
        return super().sorted()

    def insert(self, min: T, max: T, data: Any):
        return super().insert((min, max, data))

    def delete(self, min: T, max: T):
        return super().delete((min, max))

    def search(self, min: T, max: Optional[T] = None) -> Iterable[tuple[tuple[T, T], Any]]:
        # intervals are in the form [min, max)
        if max is None:
            max = min
        for el in self._root.interval_search(min, max):
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
        added = 0
        for d in data:
            tree.insert(d[0], d[1], d[2])
            added += 1
            if print_tree:
                tree.print(node_to_str=lambda x: f'[{hex(x.val[0])}, {hex(x.val[1])}); {hex(x.max_upper_value)}; {x.data}')
            assert(len(tree) == added)
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
