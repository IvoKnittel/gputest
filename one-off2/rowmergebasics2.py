import numpy as np
import math


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

def merges_(prefer_vector, vote_vector, sz4, sz):
    sequence = np.array(["x"] * sz4, dtype=np.dtype('U1'))
    for k in range(0, int(sz4/4)-1):

        for j in range(4*k, 4*(k+1)-1):

            if j < prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                    sequence[j] = "c"
                    sequence[j + 1] = "c"

        for j in range(4*k, 4*(k+1)):
            if vote_vector[j] == 0:
                sequence[j]='s'

    for k in range(0, int(sz4/4)-1):

        for j in range(4*k + 2, 4*(k+1)+2 - 1):
            if j < prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                    sequence[j] = "c"
                    sequence[j + 1] = "c"

        for j in range(4*k +2, 4*(k+1)+2):
            if vote_vector[j] == 0:
                sequence[j]='s'
    return sequence[0:sz]
