import numpy as np
from rowmergebasics import A0

def is_begin_quad(quadproducts, j):
    return quadproducts[j]=='c' and (j==0 or quadproducts[j-1] != 'c')

class A1:
    def __init__(self, in_: A0, sz):
        self.couples = np.copy(in_.couples)
        self.triples = np.copy(in_.triples)
        self.singles = np.copy(in_.singles)
        self.upchain = np.copy(in_.upchain)
        self.downchain = np.copy(in_.downchain)

    def show(self, quads, sz):
        c = np.full((6, sz), 'x')
        c[0, :] = self.couples
        c[1, :] = self.singles
        c[2, :] = self.triples
        c[3, :] = quads
        c[4, :] = self.upchain
        c[5, :] = self.downchain
        waithere=1

    # convert Quads into two couples
    def quadproducts(self, quads, sz):
        sequence = np.array(["x"] * sz, dtype=np.dtype('U1'))
        for j in range(sz):
            if quads[j]=='q':
                sequence[j] = 'c'
        return sequence

    # combine those couples with Chain items to Triples
    def quadproducts2(self, quadproducts, sz):
        sequence = np.array(quadproducts)
        for j in range(0, sz-3):
            if is_begin_quad(quadproducts, j):
                if j>0 and self.upchain[j-1] == 'u':
                    sequence[j-1:j + 2] = 'ttt'
                if j+4 < sz and self.downchain[j+4] == 'd':
                    sequence[j+2:j+5] = 'ttt'

        return sequence

    # # check chain terminal items
    # check.is_true(a.upchain[0] != 'u')
    # check.is_true(a.downchain[-1] != 'd')
    #
    # if a.upchain[1]=='u':
    #     check.is_true(a.singles[0]=='s')
    #
    # if a.downchain[-2]=='d':
    #     check.is_true(a.singles[-1]=='s')
    #
    # check.is_true(a.upchain[-3] != 'u')
    # check.is_true(a.downchain[2] != 'd')
    #
    # if a.upchain[-4]=='u':
    #     check.is_true(a.triples[-3]=='t')
    #
    # if a.downchain[3]==d':
    #     check.is_true(a.triples[2]=='t')


    # merge quadproducts2 into couples
    def couples_(self, quadproducts2, sz):
        for j in range(0, sz):
            if quadproducts2[j]=='c':
                self.couples[j] = 'c'

    # merge quadproducts2 into triples
    def triples_(self, quadproducts2, sz):
        for j in range(0, sz):
            if quadproducts2[j]=='t':
                self.triples[j] = 't'

    # ... and update UpChains
    def upchain_(self, quadproducts, quadproducts2,sz):
        for j in range(1, sz-3):
            if is_begin_quad(quadproducts, j):
                if quadproducts2[j-1] == 't':
                    self.upchain[j-1] = 'x'

    # ... and update DownChains
    def downchain_(self, quadproducts, quadproducts2,sz):
        for j in range(0, sz-3):
            if is_begin_quad(quadproducts, j):
                if j+4 < sz and quadproducts2[j+4] == 't':
                    self.downchain[j+4] = 'x'

    def apply_quadproducts(self, qp, qp2, sz):
        self.couples_(qp2, sz)
        self.triples_(qp2, sz)
        self.upchain_(qp, qp2, sz)
        self.downchain_(qp, qp2, sz)