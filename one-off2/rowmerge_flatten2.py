import numpy as np
import numpy.typing as npt
from item import Item

def init_pos_quality_array(sz):
    d_pos_quality = [('pos', int), ('quality', float)]
    arr = np.array(sz, dtype=d_pos_quality)
    arr['quality'] = -1.0
    return arr

def get_triple_quality_vector4(seq4: npt.NDArray[np.dtype('U1')], k, merged_items4: npt.NDArray[Item], items4: npt.NDArray[Item]):
    triple_quality_vector = init_pos_quality_array(len(seq4))
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

def triple_quality(sequence, items, merged_items, sz4):
    triple_quality_vector_sel = -np.ones(sz4, dtype=float)
    triple_quality_vector = init_pos_quality_array(sz4)
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

def pos_quality_vector(sequence, sz):
    sequence_with_pos = init_pos_quality_array(sz)
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
    triple_quality_vector_next = init_pos_quality_array(sz_next_row)
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
    sz = len(row)
    for s in row_hi:
        for k in range(0,sz,2):
            if s[0]==row[2*k][0]:
                row[2 * k] = s
            elif s[0]==row[2*k+1][0]:
                row[2 * k + 1] = s

    return row

def row_flatten(sequence, items, merged_items, sz4):
    triple_quality_vector = triple_quality(sequence, items, merged_items, sz4)
    triple_quality_pos_vector = pos_quality_vector(triple_quality_vector, sz4)
    tree_data=[]
    triple_quality_pos_vector_next = triple_quality_pos_vector
    szvec=[]
    sz = len(triple_quality_pos_vector)
    while sz > 1:
        triple_quality_pos_vector_this, triple_quality_vector_next = flatten_row(triple_quality_pos_vector, sz)
        tree_data.append(triple_quality_pos_vector_this)
        sz = len(triple_quality_vector_next)

    tree_data.append(triple_quality_pos_vector_next)
    row_hi = tree_data.pop()
    tree_data.reverse()
    row = init_pos_quality_array(sz4)
    for row in tree_data:
        row = flatten_insert(row,row_hi)
        row_hi = row

    return row