import pytest
import pytest_check as check
import numpy as np
from item import Item, random_items
from rowmergeinputs import I0
from rowmergebasics import A0
from quadprocessing import A1
from closechain import A2
from resolvechain import A3
from flatten import flatten_
from item import Item
# The row merge tests

def is_begin_(sequence, idx, letter):
    return sequence[idx] == letter and (idx == 0 or sequence[idx-1] != letter)

def is_begin_notx(sequence, idx):
    return sequence[idx] != 'x' and (idx == 0 or sequence[idx-1] == 'x')

def is_end(sequence, idx, letter, sz):
    return sequence[idx] == letter and (idx == sz-1 or sequence[idx+1] != letter)

def is_end_notx(sequence, idx, sz):
    return sequence[idx] != 'x' and (idx == sz-1 or sequence[idx+1] != 'x')

def is_in(sequence, letter, host_sequence, host_letter, host_length, sz, truth_value):

    for j in range(sz - host_length):
        if is_begin_(host_sequence, j, host_letter):
            if j + host_length  <= sz:
                assert (letter in sequence[j:j + host_length])==truth_value


def test_vote_cover(inputs, sz):

    k=0
    found_pair = False
    while (not found_pair) or k>50:
        items_= random_items(20)
        inputs_= I0(items_)
        vote_vector_ = inputs_.vote_vector
        # There is a pair.
        for j in range(sz - 1):
            if (vote_vector_[j]==2 and vote_vector_[j + 1] == 2):
                found_pair= True
                break

        k=k+1

    check.is_true(found_pair)

    k = 0
    found_pair = False
    while (not found_pair) or k > 50:
        items_ = random_items(20)
        inputs_ = I0(items_)
        vote_vector_ = inputs_.vote_vector
        # There is a pair.
        for j in range(sz - 1):
            if (vote_vector_[j] == 0 and vote_vector_[j + 1] == 0):
                found_pair = True
                break

        k = k + 1

    check.is_true(found_pair)

def test_cover_basics(basics, sz):
    k = 0
    found_upchain = False
    found_downchain =False
    while (not (found_upchain and found_downchain)) or k > 50:
        items_ = random_items(20)
        inputs_ = I0(items_)
        a = A0(inputs_)
        found_upchain = 'u' in a.upchain or found_upchain
        found_downchain = 'u' in a.upchain or found_downchain

        k = k + 1

    check.is_true(found_upchain and found_downchain)

    k = 0
    found_upchain2 = False
    while (not found_upchain2) or k > 50:
        items_ = random_items(20)
        inputs_ = I0(items_)
        a = A0(inputs_)
        for j in range(sz - 1):
            if a.upchain[j] == 'u' and a.upchain[j+1] == 'u':
                found_upchain2 = True
                break

        k = k + 1

    check.is_true(found_upchain2)

    k = 0
    found_checkF = False
    while (not found_checkF) or k > 50:
        items_ = random_items(20)
        inputs_ = I0(items_)
        a = A0(inputs_)
        for j in range(sz - 1):
            if inputs_.prefer_vector[j] == -1 and inputs_.prefer_vector[j+1]==1:
                check.is_true(a.quads[j] == 'q' or
                              a.triples[j] == 't' or
                              a.couples[j] == 'c' or
                              a.singles[j] == 's' or
                              a.upchain[j] == 'u' or
                              a.downchain[j] == 'd')
                check.is_true(a.quads[j+1] == 'q' or
                              a.triples[j+1] == 't' or
                              a.couples[j+1] == 'c' or
                              a.singles[j+1] == 's' or
                              a.upchain[j+1] == 'u' or
                              a.downchain[j+1] == 'd')
                found_checkF = True
                break

        k = k + 1

    check.is_true(found_checkF)
    # Check F:
    # Neighboring indices j, j + 1  def test_basics(basics, sz):
    # already belong to one of Single, Couple, Chain.
