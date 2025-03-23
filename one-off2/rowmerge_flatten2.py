import numpy as np
import numpy.typing as npt
from item import Item

def init_pos_quality_array(sz):
    d_pos_quality = [('pos', int), ('quality', float)]
    arr = np.zeros(sz, dtype=d_pos_quality)
    arr['pos'] = -1
    arr['quality'] = -2.0
    return arr

def get_triple_quality_vector4(seq4: npt.NDArray[np.dtype('U1')], triple_quality_vector, k, s, merged_items4: npt.NDArray[Item], items4: npt.NDArray[Item]):

    triple_quality_vector4 = triple_quality_vector[:,4 * k + s: 4 * (k + 1) + s]
    for j in range(0,4):
        if seq4[j] == 's':
            if j > 1:
                triple_quality_vector4[0][j] = Item(merged_items4[j-2],items4[j]).quality
            else:
                triple_quality_vector4[1][j] = Item(merged_items4[j+1], items4[j]).quality

    triple_quality_vector[:, 4 * k + s: 4 * (k + 1) + s] = triple_quality_vector4

def select_triple_quality_vector4(triple_quality_vector4):
    triple_quality_vector_sel = -2*np.ones(4, dtype=float)
    for j in range(0,4):
        if triple_quality_vector4[0,j] > 0 or triple_quality_vector4[1,j] > 0:
            if triple_quality_vector4[0,j] > triple_quality_vector4[1,j]:
                triple_quality_vector_sel[j] = -triple_quality_vector4[0,j]
            else:
                triple_quality_vector_sel[j] =  triple_quality_vector4[1,j]
    return triple_quality_vector_sel

def triple_quality(sequence, items, merged_items, sz4):
    triple_quality_vector_sel = -2*np.ones(sz4, dtype=float)
    triple_quality_vector = -2*np.ones((2,sz4), dtype=float)
    for k in range(0, int(sz4 / 4)):
        seq4 = sequence[4 * k: 4 * (k + 1)]
        merged_items4 = merged_items[1,4 * k: 4 * (k + 1)-1]
        items4 = items[4 * k: 4 * (k + 1)]
        get_triple_quality_vector4(seq4, triple_quality_vector, k, 0 , merged_items4, items4)

    for k in range(0, int(sz4 / 4) - 1):
        seq4 = sequence[4 * k + 2: 4 * (k + 1) + 2]
        merged_items4 = merged_items[1,4 * k + 2: 4 * (k + 1) + 2-1]
        items4 = items[4 * k + 2: 4 * (k + 1) + 2]
        get_triple_quality_vector4(seq4, triple_quality_vector, k, 2, merged_items4, items4)

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
    num_quads = int(np.ceil(sz / 4))
    sz_next_row = int(num_quads*2)
    triple_quality_vector_next = init_pos_quality_array(sz_next_row)
    for k in range(0,num_quads):
        seq4 = triple_quality_vector[4 * k: 4 * (k + 1)]
        j=0
        n=0
        ja= np.nan
        for j in range(0,len(seq4)):
            if seq4[j]['quality'] < -1.0:
                continue

            if seq4[j]['pos']%2==0:
                ja = j
            elif ~np.isnan(ja):
                if seq4[ja]['quality'] > seq4[j]['quality']:
                    seq4[j] = (seq4[j]['pos'], np.nan)
                else:
                    seq4[ja] = (seq4[ja]['pos'], np.nan)
                ja = np.nan
            else:
                triple_quality_vector_next[2*k+n] = seq4[j]
                seq4[j] = (seq4[j]['pos'], -2.0)
                n=n+1


        if not np.isnan(ja) and seq4[ja]['quality'] > -1.0:
            triple_quality_vector_next[2*k+n] = seq4[ja]
            seq4[ja] = (seq4[ja]['pos'], -2.0)

        triple_quality_vector[4 * k: 4 * (k + 1)] = seq4

    return triple_quality_vector_next

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
    sz_hi = len(row_hi)
    sz = len(row)
    for m in range(0,sz_hi,2):
        crt_row_hi = row_hi[ m:  m + 2]
        for k in range(0,sz,4):
            crt_row = row[k:min(k+4,sz)]
            for j in range(0,len(crt_row)):
                if crt_row_hi[0]['pos']>=0 and crt_row_hi[0]['pos']==crt_row[j]['pos']:
                    crt_row[j] = crt_row_hi[0]

                if crt_row_hi[1]['pos']>=0 and crt_row_hi[1]['pos']==crt_row[j]['pos']:
                    crt_row[j] = crt_row_hi[1]
            row[k: min(k + 4, sz)] = crt_row
    return row

def row_flatten(sequence, items, merged_items, sz4):
    triple_quality_vector = triple_quality(sequence, items, merged_items, sz4)
    triple_quality_pos_vector = pos_quality_vector(triple_quality_vector, sz4)
    tree_data=[]
    triple_quality_pos_vector_this = triple_quality_pos_vector
    sz = len(triple_quality_pos_vector)
    exit =False
    while not exit:
        triple_quality_vector_next = flatten_row(triple_quality_pos_vector_this, sz)
        tree_data.append(triple_quality_pos_vector_this)
        sz = len(triple_quality_vector_next)
        if sz==0 :
            exit=True
        elif sz==1 and triple_quality_vector_next[0]['quality'] < 0:
            exit=True
        elif sz==2 and triple_quality_vector_next[0]['quality'] < 0 and triple_quality_vector_next[1]['quality'] < 0:
            exit=True
        else:
            triple_quality_pos_vector_this = triple_quality_vector_next

    row_hi = tree_data.pop()
    tree_data.reverse()
    for row in tree_data:
        row = flatten_insert(row,row_hi)
        row_hi = row

    return row