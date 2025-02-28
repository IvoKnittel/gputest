import numpy as np
import  pytest
from item import Item, random_items
from rowmergeinputs import I0
from rowmergebasics import A0
from quadprocessing import A1
from closechain import A2
from resolvechain import A3
from flatten import flatten_


def pytest_addoption(parser):
    parser.addoption(
        "--do_display", action="store_true", default=False, help="Enable display of plots"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "do_display: mark test to enable display of plots")


@pytest.fixture(scope="module")
def items():
    return random_items(20)

@pytest.fixture(scope="module")
def sz(items):
    return len(items)

@pytest.fixture(scope="module")
def inputs(items):
    return I0(items)

@pytest.fixture(scope="module")
def basics(inputs, sz):
    b = np.full((2, sz), 0)
    b[0,:]=inputs.prefer_vector
    b[1,:]=inputs.vote_vector
    return A0(inputs)

@pytest.fixture(scope="module")
def sequences_no_quad(basics, sz):
    a = A1(basics, sz)
    a.show(basics.quads,sz)
    qp = a.quadproducts(basics.quads,sz)
    qp2 = a.quadproducts2(qp,sz)
    a.apply_quadproducts(qp, qp2,sz)
    return a

@pytest.fixture(scope="module")
def sequences_close(sequences_no_quad, sz):
    in_=sequences_no_quad
    a = A2(in_, sz)
    a.closed_upchain_(in_.upchain, in_.singles, in_.triples, sz)
    a.closed_downchain_(in_.downchain, in_.singles, in_.triples, sz)
    return a

@pytest.fixture(scope="module")
def sequences_resolved_chains(sequences_close,sz):
    a = A3(sequences_close,sz)
    forced_upchain = a.forced_upchain(sequences_close.closed_upchain, sz)
    forced_downchain = a.forced_downchain(sequences_close.closed_downchain, sz)
    a.update_singles(forced_upchain, forced_downchain, sz )
    a.update_couples(forced_upchain, forced_downchain, sz )
    a.update_triples(forced_upchain, forced_downchain, sz )
    return a

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

