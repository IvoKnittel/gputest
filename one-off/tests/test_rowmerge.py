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


@pytest.fixture(scope="module")
def items():
    return random_items(20)

@pytest.fixture(scope="module")
def sz(items):
    return len(items)

@pytest.fixture(scope="module")
def inputs(items):
    return I0(items)

### Check B:
def test_votes(inputs, sz):
    vote_vector = inputs.vote_vector
    # A Two cannot be on a terminal site.
    check.not_equal(vote_vector[0], 2)
    check.not_equal(vote_vector[-1], 2)

    # Twos are alone or in pairs.
    for j in range(sz - 2):
        if vote_vector[j] == 2:
            check.is_false(np.all(vote_vector[j + 1:j + 3] == 2))


def maybe_in_votes(inputs):
    sz=inputs.sz
    vote_vector = inputs.vote_vector
    # There is a pair.
    found_pair = False
    for j in range(sz - 1):
        found_pair = found_pair or np.all(vote_vector[j + 1:j + 2] == 2)

    return found_pair

@pytest.fixture(scope="module")
def basics(inputs, sz):
    b = np.full((2, sz), 0)
    b[0,:]=inputs.prefer_vector
    b[1,:]=inputs.vote_vector
    return A0(inputs)


# def maybe_in_basics(rawbasics, sz):
#     a=rawbasics
#     # A Quad may overlap with a Single.
#     found_single_in_quad = False
#     for j in range(sz - 3):
#         if is_begin_(a.quads, j, 'q'):
#             found_single_in_quad = found_single_in_quad or np.any(a.singles[j:j + 4] == 's')
#
#     return found_single_in_quad
#
#     # A Triple may overlap with a Single.
#     found_single_in_triple = False
#     for j in range(sz - 2):
#         if is_begin_(a.triples, j, 't'):
#             found_single_in_triple = found_single_in_triple or np.any(a.singles[j:j + 3] == 's')
#
#     return found_single_in_triple

def test_basics(basics:A0, sz):
    a=basics
    c = np.full((6, sz), 'x')
    c[0, :] = a.couples
    c[1, :] = a.singles
    c[2, :] = a.triples
    c[3, :] = a.quads
    c[4, :] = a.upchain
    c[5, :] = a.downchain

    # Singles are alone or in pairs.
    for j in range(sz - 2):
        if a.singles[j] == "s":
            check.is_false(np.all(a.singles[j + 1:j + 3] == "s"))

    # check chain terminal items
    check.is_true(a.upchain[0] != 'u')
    check.is_true(a.downchain[-1] != 'd')

    if a.upchain[1]=='u':
        check.is_true(a.singles[0]=='s')

    if a.downchain[-2]=='d':
        check.is_true(a.singles[-1]=='s')

    check.is_true(a.upchain[-3] != 'u')
    check.is_true(a.downchain[2] != 'd')

    if a.upchain[-4]=='u':
        check.is_true(a.triples[-3]=='t')

    if a.downchain[3]=='d':
        check.is_true(a.triples[2]=='t')

    # Singles, Couples, Triples, Quads, or chain chains are a complete cover.
    for j in range(sz):
        check.is_true(a.quads[j]=='q' or
                      a.triples[j]=='t' or
                      a.couples[j]=='c' or
                      a.singles[j]=='s' or
                      a.upchain[j]=='u' or
                      a.downchain[j]=='d')

    # Singles, Couples, Triples and Quads, chains are disjunct.
    is_in(a.triples, 't', a.quads, 'q', 4, sz, False)
    is_in(a.couples, 'c', a.quads, 'q', 4, sz, False)
    is_in(a.singles, 's', a.quads, 'q', 4, sz, False)
    is_in(a.upchain, 'u', a.quads, 'q', 4, sz, False)
    is_in(a.downchain, 'd', a.quads, 'q', 4, sz, False)

    is_in(a.quads, 'q', a.triples, 't', 3, sz, False)
    is_in(a.couples, 'c', a.triples, 't', 3, sz, False)
    is_in(a.singles, 's', a.triples, 't', 3, sz, False)
    is_in(a.upchain, 'u', a.triples, 't', 3, sz, False)
    is_in(a.downchain, 'd', a.triples, 't', 3, sz, False)

    is_in(a.quads, 'q', a.couples, 'c', 2, sz, False)
    is_in(a.triples, 't', a.couples, 'c', 2, sz, False)
    is_in(a.singles, 's', a.couples, 'c', 2, sz, False)
    is_in(a.upchain, 'u', a.couples, 'c', 2, sz, False)
    is_in(a.downchain, 'd', a.couples, 'c', 2, sz, False)

    is_in(a.quads, 'q', a.singles, 's', 1, sz, False)
    is_in(a.triples, 't', a.singles, 's', 1, sz, False)
    is_in(a.couples, 'c', a.singles, 's', 1, sz, False)
    is_in(a.upchain, 'u', a.singles, 's', 1, sz, False)
    is_in(a.downchain, 'd', a.singles, 's', 1, sz, False)


def maybe_test_basicsCheckF(prefer_vector, sz):
    # Check F:
    # Neighboring indices j, j + 1  def test_basics(basics, sz):
    # already belong to one of Single, Couple, Chain.
    found_prefer_m1_1 = False
    for j in range(sz - 1):
        found_prefer_m1_1 = found_prefer_m1_1 or prefer_vector[j] == -1 and prefer_vector[j+1]==1

@pytest.fixture(scope="module")
def sequences_no_quad(basics, sz):
    a = A1(basics, sz)
    a.show(basics.quads,sz)
    qp = a.quadproducts(basics.quads,sz)
    qp2 = a.quadproducts2(qp,sz)
    a.apply_quadproducts(qp, qp2,sz)
    return a

def test_sequences_noquad(sequences_no_quad, sz):
    a=sequences_no_quad
    # All items are part of Single, Couple, Triples, upChain, and downChain and are disjunct.

    c = np.full((5, sz), 'x')
    c[0, :] = a.couples
    c[1, :] = a.singles
    c[2, :] = a.triples
    c[3, :] = a.upchain
    c[4, :] = a.downchain

    is_in(a.couples, 'c', a.triples, 't', 3, sz, False)
    is_in(a.singles, 's', a.triples, 't', 3, sz, False)
    is_in(a.upchain, 'u', a.triples, 't', 3, sz, False)
    is_in(a.downchain, 'd', a.triples, 't', 3, sz, False)

    is_in(a.triples, 't', a.couples, 'c', 2, sz, False)
    is_in(a.singles, 's', a.couples, 'c', 2, sz, False)
    is_in(a.upchain, 'u', a.couples, 'c', 2, sz, False)
    is_in(a.downchain, 'd', a.couples, 'c', 2, sz, False)

    is_in(a.triples, 't', a.singles, 's', 1, sz, False)
    is_in(a.couples, 'c', a.singles, 's', 1, sz, False)
    is_in(a.upchain, 'u', a.singles, 's', 1, sz, False)
    is_in(a.downchain, 'd', a.singles, 's', 1, sz, False)

    # Singles, Couples, Triples, or chain chains are a complete cover.
    for j in range(sz):
        check.is_true(a.triples[j]=='t' or
                      a.couples[j]=='c' or
                      a.singles[j]=='s' or
                      a.upchain[j]=='u' or
                      a.downchain[j]=='d')

    # An upChain starts with a Single and ends with a Triple.

    if a.upchain[-4]=='u':
        check.is_true(a.triples[-1]=='t')

    if a.downchain[3]=='d':
        check.is_true(a.triples[0]=='t')

    for j in range(1,sz):
        if is_begin_(a.upchain, j, 'u'):
            check.is_true(a.singles[j-1] == 's')

        if is_end(a.upchain, j, 'u', sz):
            check.is_true(a.triples[j+1] == 't')

    # A downChain starts with a Triple and ends with a Single.
    for j in range(1,sz):
        if is_begin_(a.downchain, j, 'd'):
            check.is_true(a.triples[j-1] == 't')

        if is_end(a.downchain, j, 'd', sz):
            check.is_true(a.singles[j+1] == 's')

@pytest.fixture(scope="module")
def sequences_close(sequences_no_quad, sz):
    in_=sequences_no_quad
    a = A2(in_, sz)
    a.closed_upchain_(in_.upchain, in_.singles, in_.triples, sz)
    a.closed_downchain_(in_.downchain, in_.singles, in_.triples, sz)
    return a

def test_sequences_closed_chains(sequences_close, sz):
    a=sequences_close
    # Check H:
    # after creation of closedChains
    # All items are part of Single, Couple, Triples, ClosedChain and are disjunct.

    is_in(a.couples, 'c', a.triples, 't', 3, sz, False)
    is_in(a.singles, 's', a.triples, 't', 3, sz, False)
    is_in(a.closed_upchain, 'u', a.triples, 't', 3, sz, True)

    is_in(a.triples, 't', a.couples, 'c', 2, sz, False)
    is_in(a.singles, 's', a.couples, 'c', 2, sz, False)
    is_in(a.closed_upchain, 'u', a.couples, 'c', 2, sz, True)

    is_in(a.triples, 't', a.singles, 's', 1, sz, False)
    is_in(a.couples, 'c', a.singles, 's', 1, sz, False)
    is_in(a.closed_upchain, 'u', a.singles, 's', 1, sz, True)

    # ClosedUpChain and are disjunct.
    within=False
    for j in range(sz):
        if is_begin_notx(a.closed_upchain, j):
            within= True

        if is_end_notx(a.closed_upchain,j, sz):
            within = False

        if within:
            check.equal(a.singles[j], 'x')
            check.equal(a.couples[j], 'x')
            check.equal(a.triples[j], 'x')
            check.equal(a.closed_downchain[j], 'x')

    within=False
    for j in range(sz):
        if is_begin_notx(a.closed_downchain, j):
            within= True

        if is_end_notx(a.closed_downchain,j, sz):
            within = False

        if within:
            check.equal(a.singles[j], 'x')
            check.equal(a.couples[j], 'x')
            check.equal(a.triples[j], 'x')
            check.equal(a.closed_upchain[j], 'x')

    # Singles, Couples, Triples, CloseChains are a complete cover.
    for j in range(sz):
        check.is_true(a.triples[j] =='t' or
                      a.couples[j] =='c' or
                      a.singles[j] =='s' or
                      a.closed_upchain[j] != 'x' or

                      a.closed_downchain[j] != 'x')

    # ClosedUpChain begins with a single and ends with a triple.
    for j in range(sz):
        if is_begin_notx(a.closed_upchain, j):
            check.equal(a.closed_upchain[j], 's')
            check.is_true(j+1==sz or a.closed_upchain[j] == 'u')

        if is_end_notx(a.closed_upchain,j, sz):
            check.is_true(a.closed_upchain[j-1] == 't')


    # ClosedDownChain begins with a triple and ends with a single.
    for j in range(sz):
        if is_begin_notx(a.closed_upchain, j):
            check.equal(a.closed_upchain[j+1], 't')

        if is_end_notx(a.closed_upchain,j, sz):
            check.equal(a.closed_upchain[j - 1], 's')
            check.not_equal(a.closed_upchain[j - 2], 's')

    # all combinations of Single, Couple, Triples, ClosedChain are valid, i.e. translate into a possible prefer_vector.
@pytest.fixture(scope="module")
def sequences_resolved_chains(sequences_close,sz):
    a = A3(sequences_close,sz)
    forced_upchain = a.forced_upchain(sequences_close.closed_upchain, sz)
    forced_downchain = a.forced_downchain(sequences_close.closed_downchain, sz)
    a.update_singles(forced_upchain, forced_downchain, sz )
    a.update_couples(forced_upchain, forced_downchain, sz )
    a.update_triples(forced_upchain, forced_downchain, sz )
    return a



def test_sequences_resolved_chains(sequences_resolved_chains, sz):
    # Check I:
    # after resolution of closedChains
    a = sequences_resolved_chains

    # All items are part of Single, Couple, Triples, and are disjunct
    is_in(a.couples, 'c', a.triples, 't', 3, sz, False)
    is_in(a.singles, 's', a.triples, 't', 3, sz, False)
    is_in(a.triples, 't', a.couples, 'c', 2, sz, False)
    is_in(a.singles, 's', a.couples, 'c', 2, sz, False)
    is_in(a.triples, 't', a.singles, 's', 1, sz, False)
    is_in(a.couples, 'c', a.singles, 's', 1, sz, False)


    # Singles, Couples, Triples, CloseChains are a complete cover.
    for j in range(sz):
        check.is_true(a.triples[j] =='t' or
                      a.couples[j] =='c' or
                      a.singles[j] =='s')


    b = np.full((7, sz), 'x')
    b[0, :] = a.couples
    b[1, :] = a.singles
    b[2, :] = a.triples

    # Count the occurrences of 't' in a.triples and 's' in a.singles
    count_t = np.char.count(a.triples, 't').sum()
    count_s = np.char.count(a.singles, 's').sum()

    # Calculate the combined expression
    num = count_t / 3 + count_s

    # Check if the result is even
    is_even = num % 2 == 0

    # Use check.equal if it is a custom assertion function
    check.equal(is_even, True)

    # Use check.equal if it is a custom assertion function
    #check.equal(is_even, True)
    # All combinations of Single, Couple, Triples are valid.


@pytest.fixture(scope="module")
def merge_sequence(sequences_resolved_chains,sz):
    a = sequences_resolved_chains
    sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
    for j in range(sz):
        if a.singles[j] == 's':
            sequence[j]='s'
        if a.couples[j] == 'c':
            sequence[j]='c'
        if a.triples[j] == 't':
            sequence[j]='t'
    return sequence


@pytest.fixture(scope="module")
def merges_flat(merge_sequence,sz):
    sequence = merge_sequence.copy()
    flatten_(sequence,sz)
    return sequence


len_ = {
    's': 1,
    'c': 2,
    't': 3
}

@pytest.fixture(scope="module")
def merges(merges_flat,sz):
    a = merges_flat
    merges = []
    j = 0
    while j < sz:
        merges.append(a[j])
        j=j+len_[a[j]]

    return np.array(merges)


@pytest.fixture(scope="module")
def num_merges(merges,sz):
    return len(merges)

# Check J;
# The number of Merges is the number of items / 2.
# all displacements are in {-1, 0, 1}.

def is_begin(sequence, idx):
    return idx==0 or sequence[idx] != sequence[idx-1]

def margins(merges_flat, num_merges, sz):
    k = 0
    sequence_lo = np.zeros(num_merges)
    sequence_hi = np.zeros(num_merges)
    j = 0
    while j < sz:
        if merges_flat[j]=='s':
            sequence_lo[k] = j
            sequence_hi[k] = j + 1
            j = j + 1
        elif merges_flat[j]=='c':
            sequence_lo[k] = j
            sequence_hi[k] = j + 2
            j = j + 2
        elif merges_flat[j]=='t':
            sequence_lo[k] = j
            sequence_hi[k] = j + 3
            j = j + 3
        k = k + 1

    return sequence_lo, sequence_hi

def test_merges_flat(merges_flat, num_merges, sz):
    lo, hi = margins(merges_flat, num_merges, sz)
    check.equal(lo[0], 0)
    check.equal(hi[-1], 20)
    for k in range(num_merges):

        width = hi[k] - lo[k]
        check.is_true(  width >= 0)
        check.is_true( width <= 3)

        lo_ref = 2*k
        hi_ref = 2*k + 2
        displacement_lo = lo[k] - lo_ref
        displacement_hi = hi[k] - hi_ref

        check.is_true(abs(displacement_lo) <= 1)
        check.is_true(abs(displacement_hi) <= 1)

def test_num_merges(merges_flat, num_merges, sz):
    if sz % 2 == 0:
        check.equal(num_merges,sz/2)