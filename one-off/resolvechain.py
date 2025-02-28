import numpy as np

class ChainFinder:
    def __init__(self, letter):
        self.j_first = 0
        self.within = False
        self.letter= letter


    def next_chain(self, sequence, j, sz):
        if not self.within and sequence[j] == self.letter:
            self.within = True
            self.j_first = j
        if self.within and sequence[j] != self.letter:
            self.within = False
            return True, self.j_first, j -1

        if self.within and j == sz - 1:
            return True, self.j_first, j

        return False, -1, -1


class A3:
    def __init__(self, A2, sz):
        self.couples = np.copy(A2.couples)
        self.triples = np.copy(A2.triples)
        self.singles = np.copy(A2.singles)

    # If the pair is Single - Single, and at least one of the Singles has a
    # Couple as neighbor, create a Triple - Single or Single - Triple.
    # If that is not the case, the Singles are neighbors and are merged
    # into a Couple. If the pair is Triple - Triple, split one Triple
    # into a Single and a Couple to create a Triple - Single or Single - Triple.
    def forced_upchain(self, closed_upchain, sz):
        a = ChainFinder('u')
        sequence = np.copy(closed_upchain)
        for j in range(sz):
            found, j_first, j_last = a.next_chain(closed_upchain, j, sz)
            if found:
                self.process_upchain(sequence,j_first, j_last)
                a = ChainFinder('u')  # reset for next chain

        return sequence

    def forced_downchain(self, closed_downchain, sz):
        a = ChainFinder('d')
        sequence = np.copy(closed_downchain)
        for j in range(sz):
            found, j_first, j_last = a.next_chain(closed_downchain, j, sz)
            if found:
                self.process_downchain(sequence, j_first, j_last)
                a = ChainFinder('d')  # reset for next chain
        return sequence


    def process_upchain(self, sequence, j_first, j_last):
        length = 1 + j_last - j_first
        if length % 2 == 0:
            sequence[j_first:j_first+length] = 'c'
        else:
            sequence[j_first - 1:j_first + length] = 'c'

    def process_downchain(self, sequence, j_first, j_last):
        length = 1 + j_last - j_first
        if length % 2 == 0:
            sequence[j_first:j_first + length] = 'c'
        else:
            sequence[j_first:j_first + length + 1] = 'c'

    def update_singles(self, forced_upchain, forced_downchain, sz):
        for j in range(0, sz):
            if forced_upchain[j]=='s':
                self.singles[j] = 's'
            if forced_downchain[j]=='s':
                self.singles[j] = 's'

    def update_couples(self, forced_upchain, forced_downchain, sz):

        for j in range(0, sz):
            if forced_upchain[j] == 'c':
                self.couples[j] = 'c'

            if forced_downchain[j] == 'c':
                self.couples[j] = 'c'

    def update_triples(self, forced_upchain, forced_downchain, sz):
        for j in range(0, sz):
            if forced_upchain[j] == 't':
                self.triples[j] = 't'

        for j in range(0, sz):
            if forced_downchain[j] == 't':
                self.triples[j] = 't'