import pytest
import matplotlib.pyplot as plt
import cmath
import math
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


# def pytest_addoption(parser):
#     parser.addoption(
#         "--do_display", action="store_true", default=False, help="Enable display of plots"
#     )
#
# def pytest_configure(config):
#     config.addinivalue_line("markers", "do_display: mark test to enable display of plots")


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

def next_row_items(items_):
    # sz_=20
    sz_=len(items_)
    in_ = I0(items_)
    b = basics(in_, sz_)
    snq = sequences_no_quad(b, sz_)
    sc = sequences_close(snq, sz_)
    src = sequences_resolved_chains(sc, sz_)
    ms = merge_sequence(src, sz_)
    mf = merges_flat(ms, sz_)
    next_row_items_  = merges(mf, items_, sz_)

    return next_row_items_

def items_for_display(items_,to_size):
    a = [i.value() for i in items_]
    n = [i.num for i in items_]
    # s = [i.std_var() for i in m]
    pos2 = np.cumsum(n)
    k = 0
    a_to_size = np.zeros(to_size)
    x_to_size = np.zeros(to_size)
    # s_to_size= np.zeros(to_size)
    for j in range(to_size):
        if j >= pos2[k]:
            k = k + 1

        a_to_size[j] = a[k]
        x_to_size[j] = j
        # s_to_size[j] = s[k]

    return x_to_size,a_to_size # s_to_size[j]


def tree(base_items, max_gens):
    base_sz=len(base_items)
    a = [i.value() for i in base_items]
    x = [j for j in range(0, base_sz)]
    items_ = base_items
    plot_data = []
    num_gens = 0

    while num_gens <= max_gens and len(items_) < 3:
        num_gens = num_gens + 1
        items2 = next_row_items(items_)
        x2, a2 = items_for_display(items2, base_sz)
        plot_data.append([x2, a2])
        items_ = items2

    return plot_data

def tree_visual(base_items, tree_data, num_gens, start_plot_gen):
    x, a = items_for_display(base_items, len(base_items))
    plt.figure(1)
    plt.plot(x, a, 'o', color='red')
    colors = ['blue', 'green', 'black', 'cyan', 'gray', 'magenta']
    for g in range(num_gens):
        if g >= start_plot_gen:
            crt_data = tree_data[g]
            plt.plot(crt_data[0], crt_data[1]+(g-start_plot_gen)*0.02, 'o', color=colors[g-start_plot_gen])
        # plt.plot(x2, a2 + s2/10, label='a2 + s2', linestyle='--')
        # plt.plot(x2, a2 - s2/10, label='a2 - s2', linestyle='--')

    plt.xlabel('x')
    plt.ylabel('Values')
    plt.title('Plot of a, a2, and a2 ± s2 versus x')
    plt.show()


def test_tree_visual():
    base_items = sine_items()
    num_gens = 5
    start_plot_gen = 3
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)


def test_tree_visual():
    base_items = calendar_items()
    num_gens = 5
    start_plot_gen = 3
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)


def test_tree_visual():
    base_items = odd_step(20)
    num_gens = 3
    start_plot_gen = 0
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)

def test_tree_visual():
    base_items = random_items(20)
    num_gens = 3
    start_plot_gen = 0
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)
