import numpy as np

class A2:
    def __init__(self, other, sz):
        self.couples = np.copy(other.couples)
        self.triples = np.copy(other.triples)
        self.singles = np.copy(other.singles)
        self.closed_upchain = np.copy(other.upchain)
        self.closed_downchain = np.copy(other.downchain)

    # ClosedUpChain
    def closed_upchain_(self, upchain, singles ,triples, sz):
        for j in range(0, sz):
            if j>0 and upchain[j] == 'u' and singles[j-1] == 's':
                   self.closed_upchain[j - 1] = 's'
                   self.singles[j - 1] = 'x'

        for j in range(0, sz):
            if j + 4 <= sz and upchain[j] == 'u' and triples[j + 1] == 't':
                   self.closed_upchain[j + 1:j + 4] = 'ttt'
                   self.triples[j + 1:j + 4] = 'xxx'

    # ClosedDownChain
    def closed_downchain_(self, downchain, singles, triples, sz):
        for j in range(0, sz):
            if j > 2 and downchain[j] == 'd' and triples[j-1] == 't':
                self.closed_downchain[j-3:j] = 'ttt'
                self.triples[j - 3:j] = 'xxx'

        for j in range(0, sz):
            if j+1 < sz  and downchain[j] == 'd' and singles[j + 1] == 's':
                self.closed_downchain[j + 1] = 's'
                self.singles[j + 1] = 'x'