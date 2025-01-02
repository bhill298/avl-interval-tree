from typing import Any, Callable, cast, Generic, Optional, Protocol, Type, TypeVar
from avl_tree import AvlTreeNode, GenericAvlTree, T

# https://en.wikipedia.org/wiki/Interval_tree#Augmented_tree


class IntervalTreeNode(AvlTreeNode, Generic[T]):
    __slots__ = 'max_upper_value',

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_upper_value = 0

    def __str__(self):
        s = str(self.val) if self.val is not None else '<Empty>'
        return f'IntervalTreeNode({s})'

    @staticmethod
    def __callback(node, children):
        node.max_upper_value = max(i.max_upper_value for i in children + (node,))

    def _update_node_height(self, *args, **kwargs):
        return super()._update_node_height(self.__callback, *args, **kwargs)
        

class IntervalTree(GenericAvlTree):
    def __init__(self, *args, **kwargs):
        super().__init__(IntervalTreeNode, *args, **kwargs)

    @staticmethod
    def test(*args, **kwargs):
        super(IntervalTree, IntervalTree).test(IntervalTreeNode, *args, **kwargs)
        #        STRIDE = 0x1000
        ## [start, end), value
        #data = [(STRIDE * 0, STRIDE * 5, 0x0),
        #        (STRIDE * 3, STRIDE * 12, 0x5),
        #        (STRIDE * 5, STRIDE * 6, 0x1),
        #        (STRIDE * 100, STRIDE * 102, 0x2),
        #        (STRIDE * 20, STRIDE * 22, 0x3),
        #        (STRIDE * 1000, STRIDE * 1010, 0x4)]
        #tests = [
        #    # contains 0x0
        #    (STRIDE * 0, (0x0,)),
        #    # contains 0x0, 0x5
        #    (STRIDE * 4, (0x0, 0x5)),
        #    # contains 0x5, 0x1
        #    (STRIDE * 5, (0x5, 0x1)),
        #    # contains 0x2
        #    ((STRIDE * 102) - 1, (0x2,)),
        #    # contains 0x3
        #    (STRIDE * 21, (0x3,)),
        #    # contains nothing
        #    ((STRIDE * 1000) - 1, (None,)),
        #]
        #tree = cls()
        #for d in data:
        #    tree.add(d[0], d[1], d[2])
        #for test in tests:
        #    val = tree.search(test[0])
        #    if val is None:
        #        assert(test[1] is None)
        #    else:
        #        try:
        #            for i1, i2 in zip(sorted(val), sorted(test[1]), strict=True):
        #                assert(i1 == i2)
        #        except ValueError:
        #            # they are not the same length
        #            assert(False)


if __name__ == '__main__':
    IntervalTree.test()
