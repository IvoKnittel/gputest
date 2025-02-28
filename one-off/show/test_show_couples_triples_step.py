from copyreg import dispatch_table
import numpy as np
import couples_triples as c
import pytest
from items import Item, odd_step
import matplotlib.pyplot as plt


@pytest.fixture
def items():
    return odd_step(10)


@pytest.fixture
def a(items):
    lo_is_better = []
    lo_is_better.append(False)
    for idx in range(1, len(items)-1):
        Item_lo = Item(items[idx - 1], items[idx])
        Item_hi = Item(items[idx], items[idx+1])
        lo_is_better.append(Item_lo.quality > Item_hi.quality)

    lo_is_better.append(True)
    a = c.couples_chains(np.array(lo_is_better))
    a.values = np.array([item.value() for item in items])

    return a


def test_couples_chains(a):
    assert np.all(a.belongs_up_chain | a.belongs_down_chain | a.belongs_couple)


def test_force_couples(a):
    c.force_couples(a)
    # confirm there are only chains with one element left
    for current_index in range(1, a.sz):
        previous_index = current_index - 1
        assert not (a.belongs_up_chain[previous_index] and a.belongs_up_chain[current_index])
        assert not (a.belongs_down_chain[previous_index] and a.belongs_down_chain[current_index])

    # confirm each single element is attached to a couple
    for current_index in range(1, a.sz):
        if a.belongs_up_chain[current_index]:
            assert current_index < a.sz - 2
            assert a.belongs_couple[current_index + 1] and a.belongs_couple[current_index + 2]

        if a.belongs_down_chain[current_index]:
            assert current_index > 1
            assert a.belongs_couple[current_index - 1] and a.belongs_couple[current_index - 2]

    assert not a.belongs_up_chain[-1]
    assert not a.belongs_down_chain[0]


def try_a_contains_triples(a):
    c.force_couples(a)
    does_contain_triples = np.any(a.belongs_up_chain) or np.any(a.belongs_down_chain)
    return does_contain_triples


@pytest.fixture
def a_with_single_chains(a,request):
    does_contain_triples = try_a_contains_triples(a)
    while not does_contain_triples:
        does_contain_triples = try_a_contains_triples(a)

    def print_arrays():
        print("print arrays disabled")
        # print(f"lo_is_better: {lo_is_better}")
        # print(f"belongs_up_chain: {a.belongs_up_chain}")
        # print(f"belongs_down_chain: {a.belongs_down_chain}")
        # print(f"belongs_couple: {a.belongs_couple}")
        # print(f"is_couple_begin: {a.is_couple_begin}")
        # print(f"is_triple_begin: {is_triple_begin}")
        # print(f"belongs_triple: {belongs_triple}")

    request.addfinalizer(print_arrays)
    return a


def test_single_chains(a_with_single_chains):
    a = a_with_single_chains
    assert np.all(a.belongs_up_chain | a.belongs_down_chain | a.belongs_couple)


@pytest.fixture
def a_with_triples(a_with_single_chains):
    a = a_with_single_chains
    c.form_more_couples(a)
    c.form_triples(a)
    return a


def test_show_triples(a_with_triples, items):
    a = a_with_triples
    # Triple do not overlap
    for current_idx in range(0, a.sz - 2):
        if a.is_triple_begin[current_idx]:
            assert not a.is_triple_begin[current_idx + 1]
            assert not a.is_triple_begin[current_idx + 2]

    # triple_start_indices = np.where(a.is_triple_begin)[0]
    # if len(triple_start_indices) == 0:
    #    print("No triples found in the input array")
    #    return

    ## the number of triples is even
    # assert len(triple_start_indices) % 2 == 0
    ## the first triple is even
    # assert triple_start_indices[0] % 2 == 0

    print("Image code reached.")
    # Create a 2D NumPy array
    disp_array = np.zeros((4, a.sz))
    disp_array[0, :] = a.values / np.max(a.values)
    disp_array[1, :] = a.lo_is_better.astype(int)
    disp_array[2, :] = a.belongs_couple.astype(int)
    disp_array[3, :] = a.belongs_triple.astype(int)

    fig, ax = plt.subplots(figsize =[50,100])
    fig.canvas.manager.set_window_title("One-off tree row step by step")  # Set the window title

    row_labels = ['Values', 'Lo Is Better', 'Belongs to Couple', 'Belongs to Triple']
    plt.imshow(disp_array, cmap='viridis', aspect='auto')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.show()



if __name__ == "__main__":
    pytest.main(["-s"])