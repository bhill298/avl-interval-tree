from avl_tree import AvlTree

# https://stackoverflow.com/questions/17466218/what-are-the-differences-between-segment-trees-interval-trees-binary-indexed-t
# A segment tree stores intervals and is optimized for "which of these intervals contains a given point" queries.
# An interval tree stores intervals as well, but optimized for "which of these intervals overlap with a given interval" queries. It can also be used for point queries - similar to segment tree.
# segment tree https://en.wikipedia.org/wiki/Segment_tree
class SegmentTree(AvlTree):
    pass

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