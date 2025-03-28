import pytest
import matplotlib.pyplot as plt
import pytest_check as check
import numpy as np
from item import Item, random_items, odd_step, calendar_items, sine_items, chirp_items
from rowmergeinputs import I0
from rowmergebasics import A0
from quadprocessing import A1
from closechain import A2
from resolvechain import A3
from flatten import flatten_

# @pytest.fixture(autouse=True)
# def set_do_display(pytestconfig):
#     global do_display
#     do_display = pytestconfig.getoption("do_display")


def pytest_addoption(parser):
    parser.addoption(
        "--do_display", action="store_true", default=False, help="Enable display of plots"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "do_display: mark test to enable display of plots")



def items():
    return random_items(20)


def sz(items):
    return len(items)


def inputs(items):
    return I0(items)


def basics(inputs, sz):
    b = np.full((2, sz), 0)
    b[0,:]=inputs.prefer_vector
    b[1,:]=inputs.vote_vector
    return A0(inputs)


def sequences_no_quad(basics, sz):
    a = A1(basics, sz)
    a.show(basics.quads,sz)
    qp = a.quadproducts(basics.quads,sz)
    qp2 = a.quadproducts2(qp,sz)
    a.apply_quadproducts(qp, qp2,sz)
    return a


def sequences_close(sequences_no_quad, sz):
    in_=sequences_no_quad
    a = A2(in_, sz)
    a.closed_upchain_(in_.upchain, in_.singles, in_.triples, sz)
    a.closed_downchain_(in_.downchain, in_.singles, in_.triples, sz)
    return a


def sequences_resolved_chains(sequences_close,sz):
    a = A3(sequences_close,sz)
    forced_upchain = a.forced_upchain(sequences_close.closed_upchain, sz)
    forced_downchain = a.forced_downchain(sequences_close.closed_downchain, sz)
    a.update_singles(forced_upchain, forced_downchain, sz )
    a.update_couples(forced_upchain, forced_downchain, sz )
    a.update_triples(forced_upchain, forced_downchain, sz )
    return a


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



def merges_flat(merge_sequence,sz):
    sequence = merge_sequence.copy()
    flatten_(sequence,sz)
    return sequence


len_ = {
    's': 1,
    'c': 2,
    't': 3
}

def merges(merges_flat,items, sz):
    a = merges_flat
    merge_instruction = []
    merge_items=[]
    j = 0
    while j < sz:
        if a[j]=='s':
            merge_items.append(items[j])
            j = j + len_[a[j]]
            continue

        tmp_item =Item(items[j],items[j+1])
        if a[j]=='c':
            merge_items.append(tmp_item)
            j = j + len_[a[j]]
            continue

        merge_items.append(Item(tmp_item,items[j+2]))
        j = j+len_[a[j]]

    return np.array(merge_items,dtype=Item)


def num_merges(merges,sz):
    return len(merges)

def test_merges_visual():
    # sz_=20
    items_ = odd_step(20)
    sz_=len(items_)
    in_ = I0(items_)
    b = basics(in_, sz_)
    snq = sequences_no_quad(b, sz_)
    sc = sequences_close(snq, sz_)
    src = sequences_resolved_chains(sc, sz_)
    ms = merge_sequence(src, sz_)
    mf = merges_flat(ms, sz_)
    m  = merges(mf, items_, sz_)
    a = [i.value() for i in items_]
    a2 = [i.value() for i in m]
    n2 = [i.num for i in m]
    pos2 = np.cumsum(n2)
    k=0
    a3= np.zeros(sz_)
    x3= np.zeros(sz_)
    for j in range(sz_):
        if j< pos2[k]:
            a3[j]=a2[k]
            x3[j]=j
        else:
            k=k+1
            a3[j]=a2[k]
            x3[j]=j



    # s2 = [i.std_var() for i in m]
    x = [j for j in range(0, sz_)]


    # Converting lists to numpy arrays for element-wise operations
    # a2 = np.array(a2)
    # s2 = np.array(s2)

    # Plotting

    plt.figure(1)
    plt.plot(x, a, 'o', label='a')
    plt.plot(x3, a3, 'o', label='a3')
    # plt.plot(x2, a2 + s2/10, label='a2 + s2', linestyle='--')
    # plt.plot(x2, a2 - s2/10, label='a2 - s2', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Values')
    plt.title('Plot of a, a2, and a2 ± s2 versus x')
    plt.legend()
    plt.show()
    plt.legend()
    plt.show()
    waithere=1