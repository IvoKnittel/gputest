import numpy as np
import numpy.typing as npt
from item import Item

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


no_terminal = 2
undefined =  0
terminal = 4
single = 7
to_lo = 5
to_hi = 6
on_edge=8
matching=9
j_undefined = -1

d_pos_quality = [('pos', int), ('quality', np.uint16), ('kind', np.uint8)]
def init_pos_quality_array(sz):
    arr = np.zeros(sz, dtype=d_pos_quality)
    arr['pos'] = j_undefined
    arr['quality'] = undefined
    arr['kind'] = no_terminal
    return arr

def pos_quality_vector(sequence, sz):
    sequence_with_pos = init_pos_quality_array(sz)
    j=0
    for s in sequence:
       if s < -1.0:
           sequence_with_pos[j] = (j, 0, no_terminal)
       elif s >= 0.0:
            sequence_with_pos[j]=(j,np.uint16(np.abs(s) * np.iinfo(np.uint16).max), to_hi)
       else: #s < 0.0:
           sequence_with_pos[j] = (j, np.uint16(np.abs(s) * np.iinfo(np.uint16).max), to_lo)
       j=j+1

    return sequence_with_pos


def flatten_row4(seq4, seq4_next):
    n = 0
    j_begin = j_undefined

    is_inside = False
    for j in range(0, len(seq4)):
        if seq4[j]['kind'] == no_terminal:
            continue

        if seq4[j]['kind'] == to_hi or seq4[j]['kind'] == to_lo and j%2==0:
            j_begin = j
            is_inside=True
        elif is_inside:
            if seq4[j_begin]['quality'] > seq4[j]['quality']:
                seq4[j] = (seq4[j]['pos'], undefined, single)
            else:
                seq4[j_begin] = (seq4[j_begin]['pos'], undefined, single)
            is_inside = False
        elif seq4[j]['pos']!=j_undefined:
            seq4_next[n] = seq4[j]
            n = n + 1

    if is_inside:
        seq4_next[n] = seq4[j_begin]

# flatten how it works:

#   ..... s ..... s    singles
#         q       q    signed triple qualities
#       |     |     |  quads  containing 0 to 2 singles
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

def flatten_row(triple_quality_vector, sz):
    num_quads = int(np.ceil(sz / 4))
    sz_next_row = int(num_quads*2)
    triple_quality_vector_next_ = init_pos_quality_array(sz_next_row)
    for k in range(0,num_quads):
        end_idx = min(4 * (k + 1),len(triple_quality_vector))
        length=  end_idx - 4*k
        length_next = int(length/2)
        flatten_row4(triple_quality_vector[4 * k: end_idx], triple_quality_vector_next_[2 * k:2*k+length_next])

    return triple_quality_vector_next_


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
        elif sz==1 and triple_quality_vector_next[0]['pos'] == j_undefined:
            exit=True
        elif sz==2 and triple_quality_vector_next[0]['pos'] == j_undefined and triple_quality_vector_next[1]['pos'] == j_undefined:
            exit=True
        else:
            triple_quality_pos_vector_this = triple_quality_vector_next

    row_hi = tree_data.pop()
    tree_data.reverse()
    is_on_edge = False
    is_inside=False
    j=0
    j_begin = j_undefined
    for s in row_hi:
        if s['kind'] == no_terminal:
            if is_on_edge:
                row_hi[j]['kind'] = on_edge
            else:
                row_hi[j]['kind'] = matching
        else: # is terminal
            is_on_edge=True
            if j % 2 == 0:
                j_begin = j
                is_inside = True
            elif is_inside:
                if row_hi[j_begin]['quality'] > row_hi[j]['quality']:
                    row_hi[j] = (row_hi[j]['pos'],undefined, single)
                else:
                    row_hi[j_begin] = (row_hi[j_begin]['pos'],undefined, single)
                    is_inside = False
        j = j+1

    j=0
    row=[]
    for row in tree_data:
        row = flatten_insert(row,row_hi)
        row_hi = row

    return row


def flatten_insert2(crt_row_hi, crt_row):
    no_terminal_val = no_terminal
    if crt_row_hi['kind'] == on_edge:
        no_terminal_val = on_edge
    if crt_row_hi['kind'] == matching:
        no_terminal_val = matching
    j_not_assigned = j_undefined
    for j in range(0, len(crt_row)):
        if not (crt_row_hi['kind'] == to_lo or crt_row_hi['kind'] == to_hi):
            crt_row[j]['kind'] = no_terminal_val
            if no_terminal_val == no_terminal:
                j_not_assigned = j
        else:
            if crt_row_hi['pos'] == crt_row[j]['pos']:
                crt_row[j] = crt_row_hi

                if crt_row_hi['pos'] % 2 == 0:
                    no_terminal_val = matching
                else:
                    no_terminal_val = on_edge

                if j_not_assigned != j_undefined:
                    crt_row[j_not_assigned] = (crt_row[j_not_assigned]['pos'], undefined
                                               , no_terminal_val)
    return crt_row

def flatten_insert(row, row_hi):
    sz_hi = len(row_hi)
    sz = len(row)
    for m in range(0,sz_hi):
        crt_row_hi = row_hi[m]
        for k in range(0,2):
            row[2 * m + k: min(2 * m + k + 2, sz)] = flatten_insert2(crt_row_hi, row[2*m+k:min(2*m+k+2,sz)])
    return row