import numpy as np
from rowmergebasics2 import  rowmerge_no_flat
import pytest_check as check

def test_basics(inputs, items, merged_items, sz):
    sz = inputs.sz
    sequence = rowmerge_no_flat(items, merged_items, inputs.prefer_vector, inputs.vote_vector, inputs.quality_vector, sz)
    for letter in sequence:
        check.not_equal(letter , 'x')

    check.is_true(np.count_nonzero(sequence == 's') % 2 == 0)

def test_flatten(inputs, items, merged_items, sz):
    sz = inputs.sz
    sequence = rowmerge_no_flat(items, merged_items, inputs.prefer_vector, inputs.vote_vector, inputs.quality_vector, sz)
    row_flatten(triple_quality_vector, sz4)