import numpy as np
from rowmergeinputs import I0
def is_begin(sequence, idx, letter):
    return sequence[idx] == letter and (idx == 0 or sequence[idx-1] != letter)

class A0:
    def __init__(self, inputs:I0):
        sz = inputs.sz
        self.quads_(inputs.vote_vector, sz)
        self.triples_(inputs.vote_vector,sz)
        self.singles_(inputs.vote_vector,sz)
        self.couples_(inputs.prefer_vector, inputs.vote_vector, sz)
        self.upchain_(inputs.prefer_vector,inputs.vote_vector,sz)
        self.downchain_(inputs.prefer_vector,inputs.vote_vector, sz)

    def couples_(self, prefer_vector, vote_vector, sz):
        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(0, sz - 1):
            if j < sz - 1 and prefer_vector[j] == 1 and prefer_vector[j + 1] == -1:
                if vote_vector[j] == 1 and vote_vector[j+1] == 1:
                    sequence[j] = "c"
                    sequence[j+1] = "c"

        self.couples = sequence

    def quads_(self, vote_vector, sz):
        sequence0 = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(1, sz - 1):
            if vote_vector[j] == 2 and vote_vector[j + 1] == 2:
                sequence0[j] = "q"

        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(1, sz):
            if sequence0[j] == "q":
                sequence[j - 1] = "q"
                sequence[j] = "q"
                sequence[j + 1] = "q"
                sequence[j + 2] = "q"
        self.quads = sequence

    def triples_(self, vote_vector, sz):
        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(1, sz - 1):
            if vote_vector[j - 1] != 2 and vote_vector[j] == 2 and vote_vector[j + 1] != 2:
                sequence[j-1:j+2] = "ttt"

        self.triples = sequence

    def singles_(self, vote_vector, sz):
        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(0, sz):
            if vote_vector[j] == 0 and (j==0 or vote_vector[j-1] != 2) and (j== sz-1 or vote_vector[j+1] != 2):
                sequence[j] = "s"
        self.singles = sequence

    def upchain_(self, prefer_vector, vote_vector, sz):
        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(0, sz - 1):
            if (prefer_vector[j - 1] == 1
                and prefer_vector[j] == 1
                and prefer_vector[j + 1] == 1
                and vote_vector[j + 1] != 2
            ):
                sequence[j] = "u"

        self.upchain = sequence

    def downchain_(self, prefer_vector, vote_vector, sz):
        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(1, sz - 1):
            if (prefer_vector[j - 1] == -1
                and prefer_vector[j] == -1
                and prefer_vector[j + 1] == -1
                and vote_vector[j - 1] != 2
            ):
                sequence[j] = "d"

        self.downchain = sequence
