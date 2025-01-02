from collections.abc import Collection, Iterable
from typing import Any, Callable, cast, Generic, Optional, Protocol, Type, TypeVar
from avl_tree import AvlTreeNode, ComparableTreeDataType, GenericAvlTree, T

# https://en.wikipedia.org/wiki/Interval_tree#Augmented_tree


# TODO: do interval operations
# TODO: get tests passing
class IntervalTreeNode(AvlTreeNode, Generic[T]):
    __slots__ = 'max_upper_value', 'data'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val: None | tuple[T, T]
        self.data: Any = None

    @staticmethod
    def __callback(node, children):
        node.max_upper_value = max(i.max_upper_value for i in children + (node,))

    def insert(self, to_insert_val: tuple[T, T, Any], *args, **kwargs):
        new_root, node, inserted = super().insert(to_insert_val[:2], *args, **kwargs)
        node = cast(IntervalTreeNode, node)
        node.max_upper_value = cast(tuple, node.val)[1]
        # this will update the node if it already existed
        node.data = to_insert_val[2]
        return new_root, node, inserted

    def _update_node_height(self, *args, **kwargs):
        return super()._update_node_height(self.__callback, *args, **kwargs)
        

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

    def search(self, min: T, max: Optional[T] = None) -> Any:
        # intervals are in the form [min, max)
        if max is None:
            max = min
        node = cast(IntervalTreeNode, self._root.search((min, max))[0])
        if node is None:
            return None
        return node.data

    def extend(self, vals: Iterable[tuple[T, T, Any]]) -> int:
        return super().extend(vals)

    @staticmethod
    def test(*args, **kwargs):
        STRIDE = 0x1000
        # [start, end), value
        data = [(STRIDE * 0, STRIDE * 5, 0x0),
                (STRIDE * 3, STRIDE * 12, 0x5),
                (STRIDE * 5, STRIDE * 6, 0x1),
                (STRIDE * 100, STRIDE * 102, 0x2),
                (STRIDE * 20, STRIDE * 22, 0x3),
                (STRIDE * 1000, STRIDE * 1010, 0x4)]
        tests = [
            # contains 0x0
            (STRIDE * 0, (0x0,)),
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
        for test in tests:
            val = None
            val = tree.search(test[0])
            if val is None:
                assert(test[1] is None)
            else:
                try:
                    for i1, i2 in zip(sorted(val), sorted(test[1]), strict=True):
                        assert(i1 == i2)
                except ValueError:
                    # they are not the same length
                    assert(False)


if __name__ == '__main__':
    IntervalTree.test()
