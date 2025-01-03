## AVL Interval Tree

A python AVL tree implementation and an interval tree implementation that uses the AVL tree. python 3.10 or newer is
required.

### AVL Tree
An AVL tree is a form of self-balancing binary search tree. It can efficiently
store and query sortable values (e.g. integers or strings).
#### Example
```python
from avl_tree import AvlTree
AvlTree.test()
tree = AvlTree()
tree.insert(5)
tree.extend([6,7,8])
tree.delete(7)
tree.print()
# True
print(tree.search(5))
# True
print(5 in tree)
# [5, 6, 8]
print(list(tree.sorted()))
```

### Interval Tree
An interval tree lets you efficiently store and query 2D intervals or points in
the form of [min, max) (they can be extended into higher dimensions in general,
but this implementation supports 2D intervals). You can then query all
intervals that intersect a given interval or point. A point is an interval with
the same min and max. Each interval in the tree can store a piece of data
associated with it.
#### Example
```python
from interval_tree import IntervalTree
IntervalTree.test()
tree = IntervalTree()
tree.insert(1, 10, 'data')
tree.insert(5, 5, 'more data')
tree.insert(12, 16, 'some more data')
tree.insert(20, 21, 'to delete')
tree.delete(20, 21)
tree.print()
# [((5, 5), 'more data'), ((1, 10), 'data')]
print(list(tree.search(5)))
# [((5, 5), 'more data'), ((12, 16), 'some more data'), ((1, 10), 'data')]
print(list(tree.search(1,20)))
# []
print(list(tree.search(100)))
```
