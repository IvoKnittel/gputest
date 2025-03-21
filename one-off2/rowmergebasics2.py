import numpy as np
import numpy.typing as npt
from item import Item
import math


def pad4_(items_sz, merged_items_sz, prefer_vector_sz, vote_vector_sz, quality_vector_sz, sz):
    sz4 = math.ceil(sz / 4) * 4
    prefer_vector = -np.ones(sz4, dtype=int)
    vote_vector = np.ones(sz4, dtype=int)
    quality_vector = np.ones(sz4, dtype=float)
    vote_vector[-1] = 0
    prefer_vector[0:sz] = prefer_vector_sz
    vote_vector[0:sz] = vote_vector_sz
    quality_vector[0:sz] = quality_vector_sz
    if sz4 > sz:
        prefer_vector[sz] = 1
        vote_vector[sz] = 0
        quality_vector_sz[sz]=-1

    items = np.empty(sz4, dtype=Item)
    items[0:sz] = items_sz

    merged_items = np.empty((2,sz4), dtype=Item)
    merged_items[:,0:sz] = merged_items_sz

    return items, merged_items, prefer_vector, vote_vector, quality_vector, sz4


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

def convert_single_chain(seq4: npt.NDArray[np.dtype('U1')]) -> None:
    for j in range(0,4):
        if seq4[j]=='d' or seq4[j]=='u':
            seq4[j] = 's'

def merge_quad_ss(seq4: npt.NDArray[np.dtype('U1')]) -> None:
    for j in range(0,3):
        if seq4[j]=='s' and seq4[j+1]=='s':
            seq4[j] = 'c'
            seq4[j + 1] = 'e'


def get_triple_quality_vector4(seq4: npt.NDArray[np.dtype('U1')], k, merged_items4: npt.NDArray[Item], items4: npt.NDArray[Item]):
    triple_quality_vector = -np.ones((2,4), dtype=float)
    for j in range(0,3):
        if seq4[j] == 's':
            if j > 1:
                triple_quality_vector[0,j] = Item(merged_items4[j-2],items4[j]).quality
            else:
                triple_quality_vector[1, j] = Item(merged_items4[j+1], items4[j]).quality
    return triple_quality_vector

is_single = lambda x: x > 0

def select_triple_quality_vector4(triple_quality_vector4):
    triple_quality_vector_sel = -np.ones(4, dtype=float)
    for j in np.argwhere(is_single(triple_quality_vector4[0,:])).flatten():
        if triple_quality_vector4[0,j] > triple_quality_vector4[1,j]:
            triple_quality_vector_sel[j] = -triple_quality_vector4[0,j]
        else:
            triple_quality_vector_sel[j] =  triple_quality_vector4[1,j]
    return triple_quality_vector_sel


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
            seq4[0] = 's'
            seq4[1] = c_begin
            seq4[2] = c_end
            seq4[3] = 'u'
        else:
            seq4[0] = c_begin
            seq4[1] = c_end
            seq4[2] = c_begin
            seq4[3] = c_end
    elif np.all(seq4[:] == x):
        if Item(items[1], items[2]).quality > Item(items[0], items[1]).quality:
            seq4[0] = 's'
            seq4[1] = c_begin
            seq4[2] = c_end
            seq4[3] = 'u'
        else:
            seq4[0] = c_begin
            seq4[1] = c_end
            seq4[2] = c_begin
            seq4[3] = c_end
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

def merge_ss(sequence, sz4):
    for k in range(0, int(sz4 / 4)):
        seq4 = sequence[4 * k: 4 * (k + 1)]
        convert_single_chain(seq4)
        merge_quad_ss(seq4)
        sequence[4 * k: 4 * (k + 1)] = seq4

    for k in range(0, int(sz4 / 4) - 1):
        seq4 = sequence[4 * k + 2: 4 * (k + 1) + 2]
        convert_single_chain(seq4)
        merge_quad_ss(seq4)
        sequence[4 * k +2 : 4 * (k + 1) + 2] = seq4

def triple_quality(sequence, items, merged_items, sz4):
    triple_quality_vector = -np.ones((2,sz4), dtype=float)
    triple_quality_vector_sel = -np.ones(sz4, dtype=float)
    for k in range(0, int(sz4 / 4)):
        seq4 = sequence[4 * k: 4 * (k + 1)]
        merged_items4 = merged_items[1,4 * k: 4 * (k + 1)-1]
        items4 = items[4 * k: 4 * (k + 1)]
        triple_quality_vector[:,4 * k: 4 * (k + 1)] = get_triple_quality_vector4(seq4, merged_items4, items4)

    for k in range(0, int(sz4 / 4) - 1):
        seq4 = sequence[4 * k + 2: 4 * (k + 1) + 2]
        merged_items4 = merged_items[1,4 * k + 2: 4 * (k + 1) + 2-1]
        items4 = items[4 * k + 2: 4 * (k + 1) + 2]
        triple_quality_vector[:, 4 * k + 2: 4 * (k + 1) + 2]= get_triple_quality_vector4(seq4, merged_items4, items4)

    for k in range(0, int(sz4 / 4)):
        triple_quality_vector_sel[4 * k: 4 * (k + 1)] = select_triple_quality_vector4(triple_quality_vector[:, 4 * k: 4 * (k + 1)])

    return triple_quality_vector_sel

def pos_quality_vector(sequence, sz4):
    d_pos_quality = [('pos', int), ('quality', float)]
    sequence_with_pos = np.zeros(sz4, dtype=d_pos_quality)
    j=0
    for s in sequence:
       sequence_with_pos[j]=(j,s)
       j=j+1
    return sequence_with_pos

# flatten how it works:

#   ..... s ..... s    singles
#         q       q    signed triple qualities
#       |     |     |  quads  containing 0 to 2 singles
def flatten_row(triple_quality_vector, sz):
    num_quads = np.ceil(sz / 4)
    sz_next_row = num_quads*2
    d_pos_quality = [('pos', int), ('quality', float)]
    triple_quality_vector_next = -np.ones(sz_next_row, dtype=d_pos_quality)
    for k in range(0,num_quads):
        seq4 = triple_quality_vector[4 * k: 4 * (k + 1)]
        j=0
        n=0
        ja= np.nan
        for j in range(0,4):
            if seq4[j][1] < 0:
                continue
            if j%2==0:
                ja = j
            elif ~np.isnan(ja):
                if seq4[ja][1] > seq4[j][1]:
                    seq4[j] = (seq4[j][0], np.nan)
                else:
                    seq4[ja] = (seq4[ja][0], np.nan)
                ja = np.nan
            else:
                triple_quality_vector_next[2*k+n] = seq4[j]
                n=n+1

        triple_quality_vector[4 * k: 4 * (k + 1)] = seq4

    return triple_quality_vector, triple_quality_vector_next

#  for each s store sign(j),q
#   | | |              bins , two items, or three, if it is the last
#  1st gen
#    []
#    j0,q_j0
#    j0,q_j0 j1,q_j1
#    j0,q_j0 j1,q_j1,j2,q_j2
#    j0,q_j0 j1,q_j1,j2,q_j2 j3,q_j3
#
# in each quad or bin
# from left to right, take the first even index single and pair it with
# the subsequent single.
#
# Pairing: set the lower-quality item to some dummy
# value, keep the items in the row, transfer
# the unpaired items to the next row (there can be at most two)
# once the row size==2 or 3, all singles are paired.
# now write back into the previous row, with each iten into it respective bin.

def flatten_insert(row, row_hi):
    for s in row_hi:

    return row


def rowmerge_no_flat(items_sz, merged_items_sz, prefer_vector_sz, vote_vector_sz, quality_vector_sz, sz):
    items, merged_items, prefer_vector, vote_vector, quality_vector, sz4 = pad4_(items_sz, merged_items_sz, prefer_vector_sz, vote_vector_sz, quality_vector_sz, sz)
    sequence = merges_wo_chains(prefer_vector, vote_vector, sz4, sz)

    merges_insert_chains(sequence, prefer_vector, vote_vector, sz4, sz)
    merges_resolve_chains(sequence, items, sz4)
    merge_ss(sequence, sz4)
    triple_quality_vector = triple_quality(sequence, items, merged_items, sz4)
    triple_quality_pos_vector = pos_quality_vector(triple_quality_vector, sz4)
    tree_data=[]
    triple_quality_pos_vector_next = triple_quality_pos_vector
    sz = len(triple_quality_pos_vector)
    while sz > 1:
        triple_quality_pos_vector_this, triple_quality_vector_next = flatten_row(triple_quality_pos_vector, sz)
        tree_data.append(triple_quality_pos_vector_this)
        sz = len(triple_quality_vector_next)

    tree_data.append(triple_quality_pos_vector_next)
    row_hi = tree_data.pop()
    for row in tree_data.reverse():
        row = flatten_insert(row,row_hi)
        row_hi = row

    return unpad4_(sequence, sz)