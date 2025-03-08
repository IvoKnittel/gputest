import numpy as np
import math

from cuda.opencv.opencv.modules.dnn.misc.quantize_face_detector import dtype


def pad4_(prefer_vector_sz, vote_vector_sz, sz):
    sz4 = math.ceil((sz+2) / 4)*4
    prefer_vector = -np.ones(sz4, dtype = int)
    vote_vector = np.ones( sz4, dtype = int)
    vote_vector[-1] = 0
    prefer_vector[0:sz] = prefer_vector_sz
    vote_vector[0:sz]   = vote_vector_sz
    if sz4 > sz:
        prefer_vector[sz] = 1
        vote_vector[sz] = 0

    return prefer_vector, vote_vector

def merges_wo_chains(prefer_vector, vote_vector, sz4, sz):
    sequence = np.array(["x"] * sz4, dtype=np.dtype('U1'))
    for k in range(0, int(sz4/4)-1):

        for j in range(4*k, 4*(k+1)-1):

            if prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                    sequence[j] = "c"
                    sequence[j + 1] = "c"

        for j in range(4*k, 4*(k+1)):
            if vote_vector[j] == 0:
                sequence[j]='s'


    for k in range(0, int(sz4/4)-1):

        for j in range(4*k + 2, 4*(k+1)+2 - 1):
            if prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                    sequence[j] = "c"
                    sequence[j + 1] = "c"

        for j in range(4*k +2, 4*(k+1)+2):
            if vote_vector[j] == 0:
                sequence[j]='s'

def merges_insert_chains(sequence, prefer_vector, vote_vector, sz4, sz):
    for k in range(0, int(sz4 / 4) - 1):

        for j in range(4 * k, 4 * (k + 1)):

            if vote_vector[j] == 1 and sequence[j] != "c":
                if prefer_vector[j] == 1:
                    sequence[j] = "u"
                elif prefer_vector[j] == -1:
                    sequence[j] = "d"

    for k in range(0, int(sz4 / 4) - 1):

        for j in range(4 * k + 2, 4 * (k + 1) + 2):

            if vote_vector[j] == 1 and sequence[j] != "c":
                if prefer_vector[j] == 1:
                    sequence[j] = "u"
                elif prefer_vector[j] == -1:
                    sequence[j] = "d"

    return sequence[0:sz]

def get_quad_sequence(seq4):

    count_u = 0
    count_d = 0
    for item in seq4:
        if item == 'u':
            count_u = count_u + 1
        if item == 'd':
            count_d = count_d + 1

    if count_d > count_u:
        letter = 'd'
        seq4_ = seq4[::-1]
    else:
        letter = 'u'
        seq4_ = seq4

    return seq4_, letter


def set_quad_sequence(seq4: np.array, letter):
    if letter=='d':
        seq4[:] = seq4[::-1]

def merges_resolve_quad(s:np.array, x):

 # . . x x  do nothing
 # . x x .  convert inner to couple
 # . x x x  convert into s c e x , or into c e c e
 # x x x x  convert into x c e s , or into c e c e

 def fu(items, s)
     b0 = [True,True, False,False]
     b1 = [False, True, True, False]
     b2 = [False,True, True,True]
     b3 = [True, True, True,True]
     if np.all(s[b0] == x) and np.all(s[~np.array(b0)] != x):
         s[0] = 'c'
         s[1] = 'e'
     elif np.all(s[b1] == x) and np.all(s[~np.array(b1)] != x):
         s[1] = 'c'
         s[2] = 'e'
     elif np.all(s[b2] == x) and np.all(s[~np.array(b2)] != x):
         s[1] = 'c'
         s[2] = 'e'
         s[3] = 'c'
         s[4] = 'e'
     elif np.all(s[:] == x):
         s[1] = 'c'
         s[2] = 'e'
         s[3] = 'c'
         s[4] = 'e'

 def test_fu():
     x='x'
     s = ['x','x','x','d'] # want fu(s,x) to be false
     s = ['x','x','d','d'] # want fu(s,x) to be true
     s = ['x','d','d','d'] # want fu(s,x) to be false
     s = ['x','d','x','d'] # want fu(s,x) to be false


    seq4=seq4

def merges_resolve_isolated_chain_elements(seq4):
    seq4 = np.char.replace(seq4, 'd', 's')
    seq4 = np.char.replace(seq4, 'u', 's')
    return seq4

def merges_resolve_chains(sequence, sz4):
    for k in range(0, int(sz4 / 4) - 1):
        seq4 = sequence[4 * k: 4 * (k + 1)]
        seq4_, letter =  get_quad_sequence(seq4)
        merges_resolve_quad(seq4_)
        set_quad_sequence(seq4_, letter)
        sequence[4 * k: 4 * (k + 1)]= seq4_

    for k in range(0, int(sz4 / 4) - 1):
        seq4 = sequence[4 * k + 2: 4 * (k + 1) + 2 - 1]
        seq4_, letter =  get_quad_sequence(seq4)
        merges_resolve_quad(seq4_)
        seq4_=merges_resolve_isolated_chain_elements(seq4_)
        set_quad_sequence(seq4_, letter)
        sequence[4 * k: 4 * (k + 1)]= seq4_
