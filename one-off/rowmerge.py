import numpy as np
from rowmergebasics import A0
from quadprocessing import A1
from closechain import A2
from resolvechain import A3
from flatten import flatten_
from item import Item


def rawbasics(prefer_vector, vote_vector, sz):
    a = A0()
    a.couples_(prefer_vector, sz)
    a.quads_(vote_vector,sz)
    a.triples_(vote_vector, sz)
    a.singles_(vote_vector, sz)
    a.upchain_(prefer_vector, sz)
    a.downchain_(prefer_vector, sz)
    return a

def basics(rawbasics,sz):
    a = A0(rawbasics)
    a.couples=a.overwrite_couples(sz)
    a.triples=a.overwrite_triples(sz)
    a.singles = a.overwrite_singles(sz)
    return a


def allmerges(basics, sz):
    a=basics
    sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
    for j in range(0, sz):
        if a.couples[j] == "c":
            sequence[j] = "c"
        if a.singles[j] == "s":
            sequence[j] = "s"

    for j in range(0, sz):
        if a.quads[j] == "q":
            sequence[j] = "q"
        if a.triples[j] == "t":
            sequence[j] = "t"
    return sequence


def sequences_no_quad(basics, sz):
    a = A1b(basics)
    quadproducts = a.quadproducts(sz)
    quadproducts2= a.quadproducts2(quadproducts, sz)
    a.couples_(quadproducts2, sz)
    a.triples_(quadproducts2, sz)
    a.upchain_(quadproducts2,sz)
    a.downchain_(quadproducts2,sz)
    return a


def sequences_closed_chains(sequences_no_quad, sz):
    a = A2(sequences_no_quad, sz)
    a.closed_upchain_(sequences_no_quad.upchain, sz)
    a.singles_u(sz)
    a.triples_u(sz)
    a.closed_downchain_(sequences_no_quad.downchain, sz)
    a.singles_d(sz)
    a.triples_d(sz)
    return a


def sequences_resolved_chains(sequences_closed_chains,sz):
    a = A3(sequences_closed_chains,sz)
    forced_upchain = a.forced_upchain(sequences_closed_chains.closed_upchain, sz)
    forced_downchain = a.forced_downchain(sequences_closed_chains.closed_downchain, sz)
    a.update_singles(forced_upchain, forced_downchain, sz )
    a.update_couples(forced_upchain, forced_downchain, sz )
    a.update_triples(forced_upchain, forced_downchain, sz )
    return a


def merges(sequences_closed_chains, sz):
    a = sequences_closed_chains
    sequence = a.couples
    for j in range(sz):
        if a.singles[j] =='s':
            sequence[j] = 's'
        if a.triples[j] =='t':
            sequence[j] = 't'


def merges_flat(merges, sz):
    flatten_(merges,sz)


def num_merges(merges_flat:np.array, sz):
    k = 0
    for j in range(sz):
        if is_begin(merges_flat, j):
            k = k + 1

    return k


def is_begin(sequence, idx):
    return idx==0 or sequence[idx] != sequence[idx-1]


def merged_items(items, merges_flat, num_merges, sz):
    items = np.array(num_merges, dtype=Item)
    k = 0
    for j in range(sz):
        if is_begin(merges_flat, j):
            if merges_flat[j]=='s':
                items[k] = items[j]
            elif merges_flat[j]=='c':
                items[k] = items[j:j+2]
            elif merges_flat[j]=='t':
                items[k] = items[j:j+3]
            k = k + 1

    return items