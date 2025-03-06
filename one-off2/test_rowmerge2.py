import math
from rowmergebasics2 import  pad4_, merges_


def test_basics(inputs, sz):
    sz = inputs.sz
    sz4 = math.ceil((sz + 2) / 4) * 4
    prefer_vector_, vote_vector_ = pad4_(inputs.prefer_vector, inputs.vote_vector, sz)
    raw_merges = merges_(prefer_vector_, vote_vector_, sz4, sz)