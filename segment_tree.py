from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar

from avl_tree import AvlTree

class ComparableTreeDataType(Protocol):
    @abstractmethod
    def __lt__(self, other: Any, /) -> bool: ...


T = TypeVar('T', bound=ComparableTreeDataType)

# https://stackoverflow.com/questions/17466218/what-are-the-differences-between-segment-trees-interval-trees-binary-indexed-t
# A segment tree stores intervals and is optimized for "which of these intervals contains a given point" queries.
# An interval tree stores intervals as well, but optimized for "which of these intervals overlap with a given interval" queries. It can also be used for point queries - similar to segment tree.
# segment tree https://en.wikipedia.org/wiki/Segment_tree
class SegmentTree(Generic[T]):
    __slots__ = '_tree'

    def __init__(self):
        self._tree = AvlTree()

    def __str__(self):
        # TODO:
        return ''

    def __repr__(self):
        return str(self)

    def __contains__(self, x: T):
        return self.search(x) is not None

    def __eq__(self, other):
        if not isinstance(other, SegmentTree):
            return False
        return self._tree == other._tree

    def __len__(self):
        # TODO:
        return 0
    
    def add(self, start: T, end: T, val: Any) -> bool:
        # TODO:
        return False

    def search(self, val: T) -> None | list[Any]:
        # TODO:
        return None

    @staticmethod
    def test():
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
        tree = SegmentTree()
        for d in data:
            tree.add(d[0], d[1], d[2])
        for test in tests:
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
    SegmentTree.test()
