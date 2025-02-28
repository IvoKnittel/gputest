import numpy as np

len_ ={'s':1,
       'c':2,
       't':3}

other_letter = {'s':'t','t':'s'}

class ChainFinder:
    def __init__(self):
        self.j_first = 0
        self.j_next = 0
        self.within = False
        self.start_letter='x'
        self.export = False


    def next_chain(self, sequence, j, sz):
        if not self.within:
            if sequence[j] in ['s', 't']:
                # next chain begins
                self.within = True
                self.j_first = j
                self.j_next=j+len_[sequence[j]]
                self.start_letter = sequence[j]
                return self.j_next, False, -1
            else:
                self.j_next = j + len_[sequence[j]]
                return self.j_next, False, -1

        if self.within:
            if sequence[j] in ['s', 't']:
                self.within = False
            self.j_next = j + len_[sequence[j]]
            return self.j_next, True, self.j_first


def flatten_(sequence, sz):
    a = ChainFinder()
    j = 0
    while j < sz:
        j_next, found, j_first = a.next_chain(sequence, j, sz)
        if found:
            flatten_section_simple(sequence, j_first, j)
        j = j_next

def flatten_section_simple(sequence : np.array, j_first, j):

    if sequence[j_first] != sequence[j]:
        return

    if sequence[j_first] == 's':
        # simple, quality is not considered
        if sequence[j_first+1] == 's':
            sequence[j_first:j_first + 2] = 'c'
        else:
            # simple, quality is not considered
            sequence[j_first:j_first + 3] = 't'

    if sequence[j_first] == 't':
        # simple, quality is not considered
        sequence[j_first] = 's'
        sequence[j_first+1:j_first + 3] = 'c'