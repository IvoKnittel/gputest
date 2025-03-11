
from rowmergebasics2 import  rowmerge_no_flat
import pytest_check as check
from item import random_items
def test_basics(inputs, sz):
    sz = inputs.sz
    items_=random_items(20)
    sequence = rowmerge_no_flat(items_, inputs.prefer_vector, inputs.vote_vector, sz)
    for letter in sequence:
        check.not_equal(letter , 'x')