import numpy as np
import couples_triples as c
import pytest
from items import Item

@pytest.fixture
def items():
    values = np.random.rand(30)
    itemlist=[]
    for val in values:
        itemlist.append(Item(val))
        

    return np.array(itemlist)


@pytest.fixture
def lo_is_better(items):
    values = np.random.rand(30)
    lo_is_better=[]
    lo_is_better.append(False)
    for idx in range(1,len(items)):
        Item_lo=Item(items[idx-1], items[idx])
        Item_hi=Item(items[idx-1],items[idx])
        lo_is_better.append(Item_lo.quality > Item_hi.quality)
 
    lo_is_better.append(True)
    return np.array(lo_is_better)



@pytest.fixture
def a(request, lo_is_better):
    lo_is_better = np.random.choice([False, True], size=30)
    lo_is_better[0] = False  # First element is False
    lo_is_better[-1] = True  # Last element is True
    a = c.couples_chains(lo_is_better)

    def print_arrays():
        print(f"lo_is_better: {lo_is_better}")
        print(f"belongs_up_chain: {a.belongs_up_chain}")
        print(f"belongs_down_chain: {a.belongs_down_chain}")
        print(f"belongs_couple: {a.belongs_couple}")
        print(f"is_couple_begin: {a.is_couple_begin}")
        # print(f"is_triple_begin: {is_triple_begin}")
        # print(f"belongs_triple: {belongs_triple}")

    request.addfinalizer(print_arrays)
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

def try_a_contains_triples():
    lo_is_better = np.random.choice([False, True], size=30)
    lo_is_better[0] = False  # First element is False
    lo_is_better[-1] = True  # Last element is True
    a = c.couples_chains(lo_is_better)
    c.force_couples(a)
    does_contain_triples = np.any(a.belongs_up_chain) or np.any(a.belongs_down_chain)
    return a, does_contain_triples

@pytest.fixture
def a_with_single_chains(request):
    a, does_contain_triples = try_a_contains_triples()
    while not does_contain_triples:
        a, does_contain_triples = try_a_contains_triples()
    
    def print_arrays():
        print(f"lo_is_better: {a.lo_is_better}")
        print(f"belongs_up_chain: {a.belongs_up_chain}")
        print(f"belongs_down_chain: {a.belongs_down_chain}")
        print(f"belongs_couple: {a.belongs_couple}")
        print(f"is_couple_begin: {a.is_couple_begin}")
        print(f"is_triple_begin: {a.is_triple_begin}")
        # print(f"belongs_triple: {belongs_triple}")

    request.addfinalizer(print_arrays)
    return a

def test_single_chains(a_with_single_chains):
    a=a_with_single_chains
    assert np.all(a.belongs_up_chain | a.belongs_down_chain | a.belongs_couple)

@pytest.fixture
def a_with_triples(a_with_single_chains):
    a=a_with_single_chains
    c.form_more_couples(a)
    c.form_triples(a)
    return a



def test_triples(a_with_triples):
    a=a_with_triples
    #Triple do not overlap
    for current_index in range(0, a.sz-2):
        if a.is_triple_begin[current_index]:
            assert not a.is_triple_begin[current_idx+1]
            assert not a.is_triple_begin[current_idx+2]

    triple_start_indices = np.where(a.is_triple_begin)[0]
    if len(triple_start_indices) == 0:
        print("No triples found in the input array")
        return

    # the number of triples is even
    assert len(triple_start_indices) % 2 == 0
    # the first triple is even
    assert triple_start_indices[0] % 2 == 0

#@pytest.fixture
#def eval_triples(a_with_triples, items):
#    d = a_with_triples



if __name__ == "__main__":
    pytest.main()
