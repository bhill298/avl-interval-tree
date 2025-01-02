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

# TODO: test fails when there is a false match on the exclusive end of an interval
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
        if i1[0] == i1[1] or i2[0] == i2[1]:
            # one or both of the intervals is a point (as in start and end are equal)
            return i1[0] <= i2[1] and i1[1] >= i2[0]
        else:
            # interval
            return i1[0] < i2[1] and i1[1] > i2[0]

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
            if min >= node_max_upper_value:
                # the start of this interval is greater than the end of this node and all children
                continue
            if node.left is not None:
                # search left children
                to_search.append(node.left)
            if self.__overlapswith((min, max), node.val):
                # intervals / points overlap
                yield node
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
    def test():
        STRIDE = 0x1000
        # [start, end), value
        data = [(STRIDE * 0, STRIDE * 5, 0x0),
                (STRIDE * 3, STRIDE * 12, 0x5),
                (STRIDE * 3, STRIDE * 4, 0x20),
                (STRIDE * 5, STRIDE * 6, 0x1),
                (STRIDE * 100, STRIDE * 102, 0x2),
                (STRIDE * 20, STRIDE * 22, 0x3),
                (STRIDE * 1000, STRIDE * 1010, 0x4)]
        tests = [
            # contains 0x0
            (STRIDE * 0, (0x0,)),
            # contains 0x0, 0x5, 0x20
            (int(STRIDE * 3.5), (0x0, 0x5, 0x20)),
            # contains 0x0, 0x5
            (STRIDE * 4, (0x0, 0x5)),
            # contains 0x5, 0x1
            (STRIDE * 5, (0x5, 0x1)),
            # contains 0x2
            ((STRIDE * 102) - 1, (0x2,)),
            # contains 0x3
            (STRIDE * 21, (0x3,)),
            # contains nothing
            ((STRIDE * 1000) - 1, (None,)),
        ]
        tree = IntervalTree()
        for d in data:
            tree.insert(d[0], d[1], d[2])
            tree.print(node_to_str=lambda x: f'[{hex(x.val[0])}, {hex(x.val[1])}); {hex(x.max_upper_value)}; {x.data}')
        for test in tests:
            val = None
            vals = list(tree.search(test[0]))
            print(hex(test[0]))
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


if __name__ == '__main__':
    IntervalTree.test()
