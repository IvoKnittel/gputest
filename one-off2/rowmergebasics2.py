import numpy as np
import numpy.typing as npt
from item import Item
import math


def pad4_(items_sz, prefer_vector_sz, vote_vector_sz, sz):
    sz4 = math.ceil(sz / 4) * 4
    prefer_vector = -np.ones(sz4, dtype=int)
    vote_vector = np.ones(sz4, dtype=int)
    vote_vector[-1] = 0
    prefer_vector[0:sz] = prefer_vector_sz
    vote_vector[0:sz] = vote_vector_sz
    if sz4 > sz:
        prefer_vector[sz] = 1
        vote_vector[sz] = 0

    items = np.empty(sz4, dtype=Item)
    items[0:sz] = items_sz

    return items, prefer_vector, vote_vector, sz4


def unpad4_(sequence, sz):
    return sequence[0:sz]


def merges_wo_chains(prefer_vector, vote_vector, sz4, sz):
    sequence = np.array(["x"] * sz4, dtype=np.dtype('U1'))
    for k in range(0, int(sz4 / 4)):

        for j in range(4 * k, 4 * (k + 1) - 1):

            if prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                sequence[j] = "c"
                sequence[j + 1] = "e"

        for j in range(4 * k, 4 * (k + 1)):
            if vote_vector[j] == 0:
                sequence[j] = 's'

    for k in range(0, int(sz4 / 4) - 1):

        for j in range(4 * k + 2, 4 * (k + 1) + 2 - 1):
            if prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                sequence[j] = "c"
                sequence[j + 1] = "e"

        for j in range(4 * k + 2, 4 * (k + 1) + 2):
            if vote_vector[j] == 0:
                sequence[j] = 's'

    return sequence


def merges_insert_chains(sequence, prefer_vector, vote_vector, sz4, sz):
    for k in range(0, int(sz4 / 4)):

        for j in range(4 * k, 4 * (k + 1)):

            if vote_vector[j] == 1 and sequence[j] != "c" and sequence[j] != "e":
                if prefer_vector[j] == 1:
                    sequence[j] = "u"
                elif prefer_vector[j] == -1:
                    sequence[j] = "d"

    for k in range(0, int(sz4 / 4) - 1):

        for j in range(4 * k + 2, 4 * (k + 1) + 2):

            if vote_vector[j] == 1 and sequence[j] != "c" and sequence[j] != "e":
                if prefer_vector[j] == 1:
                    sequence[j] = "u"
                elif prefer_vector[j] == -1:
                    sequence[j] = "d"

    return sequence[0:sz]


def set_quad_sequence(seq4: np.array, letter: str) -> None:
    if letter == 'd':
        seq4[:] = seq4[::-1]


def get_quad_sequence(seq4):
    count_u = 0
    count_d = 0
    for item in seq4:
        if item == 'u':
            count_u += 1
        if item == 'd':
            count_d += 1

    if count_d > count_u:
        letter = 'd'
        seq4_ = seq4[::-1]
    else:
        letter = 'u'
        seq4_ = seq4

    return seq4_, letter


def merges_resolve_quad(seq4: npt.NDArray[np.dtype('U1')],
                        items: npt.NDArray[Item],
                        x: np.dtype('U1')) -> None:

    b0 = [True, True, False, False]
    b1 = [False, True, True, False]
    b2 = [False, True, True, True]
    b3 = [False, True, False, False]
    b4 = [False, False,True, False]
    if x=='u':
        c_begin='c'
        c_end='e'
    else:
        c_begin='e'
        c_end='c'
    if np.all(seq4[b0] == x) and np.all(seq4[~np.array(b0)] != x):
        #  u u . .  convert to couple
        seq4[0] = c_begin
        seq4[1] = c_end
    elif np.all(seq4[b1] == x) and seq4[3] != x:
        #  x u u .  convert to couple
        seq4[1] = c_begin
        seq4[2] = c_end
    elif np.all(seq4[b2] == x) and np.all(seq4[~np.array(b2)] != x):
        #  . u u u  convert into s c e u , or into c e c e
        if Item(items[1], items[2]).quality > Item(items[0], items[1]).quality:
            seq4[1] = 's'
            seq4[2] = c_begin
            seq4[3] = c_end
            seq4[4] = 'u'
        else:
            seq4[1] = c_begin
            seq4[2] = c_end
            seq4[3] = c_begin
            seq4[4] = c_end
    elif np.all(seq4[:] == x):
        if Item(items[1], items[2]).quality > Item(items[0], items[1]).quality:
            seq4[1] = 's'
            seq4[2] = c_begin
            seq4[3] = c_end
            seq4[4] = 'u'
        else:
            seq4[1] = c_begin
            seq4[2] = c_end
            seq4[3] = c_begin
            seq4[4] = c_end
    elif seq4[1] == x and np.all(seq4[~np.array(b3)] != x):
        if seq4[2]=='s':
            if Item(items[0], items[1]).quality > Item(items[1], items[2]).quality:
                seq4[0] = c_begin
                seq4[1] = c_end
            else:
                seq4[1] = c_begin
                seq4[2] = c_end
    elif seq4[2] == x and np.all(seq4[~np.array(b4)] != x):
        if seq4[3]=='s':
            if Item(items[1], items[2]).quality > Item(items[2], items[3]).quality:
                seq4[1] = c_begin
                seq4[2] = c_end
            else:
                seq4[2] = c_begin
                seq4[3] = c_end

def merges_resolve_isolated_chain_elements(seq4):
    seq4 = np.char.replace(seq4, 'd', 's')
    seq4 = np.char.replace(seq4, 'u', 's')
    return seq4


def merges_resolve_chains(sequence, items, sz4):
    for k in range(0, int(sz4 / 4)):
        seq4 = sequence[4 * k: 4 * (k + 1)]
        seq4_, letter = get_quad_sequence(seq4)
        merges_resolve_quad(seq4_, items, letter)
        set_quad_sequence(seq4_, letter)
        sequence[4 * k: 4 * (k + 1)] = seq4_

    for k in range(0, int(sz4 / 4) - 1):
        seq4 = sequence[4 * k + 2: 4 * (k + 1) + 2]
        seq4_, letter = get_quad_sequence(seq4)
        merges_resolve_quad(seq4_, items, letter)
        seq4_ = merges_resolve_isolated_chain_elements(seq4_)
        set_quad_sequence(seq4_, letter)
        sequence[4 * k +2 : 4 * (k + 1) + 2] = seq4_


def rowmerge_no_flat(items_sz, prefer_vector_sz, vote_vector_sz, sz):
    items, prefer_vector, vote_vector, sz4 = pad4_(items_sz, prefer_vector_sz, vote_vector_sz, sz)
    sequence = merges_wo_chains(prefer_vector, vote_vector, sz4, sz)

    merges_insert_chains(sequence, prefer_vector, vote_vector, sz4, sz)
    merges_resolve_chains(sequence, items, sz4)
    return unpad4_(sequence, sz)