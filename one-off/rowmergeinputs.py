import numpy as np
from item import Item

def merged_items(items, sz):
    if sz < 2:
        raise ValueError("The input list must contain at least two items.")

    m = np.empty((2, sz), dtype=Item)

    m[0, 0] = Item()
    m[1, 0] = Item(items[0], items[1])

    for idx in range(1, sz - 1):
        m[0, idx] = Item(items[idx - 1], items[idx])
        m[1, idx] = Item(items[idx], items[idx + 1])

    m[0, -1] = Item(items[-2], items[-1])
    m[1, -1] = Item()

    return m


def prefer_vector(merged_items, sz):
    plist = []
    plist.append(1)
    for idx in range(1, sz-1):
        Item_lo = merged_items[0,idx]
        Item_hi = merged_items[1,idx]
        if Item_hi.quality > Item_lo.quality:
            prefer = 1
        else:
            prefer = -1
        plist.append(prefer)
    plist.append(-1)
    return np.array(plist)


def vote_vector(prefer_vector, sz):
    vvec= np.zeros(sz,dtype=int)
    for idx in range(0, sz):
        if (idx - 1) >= 0 and prefer_vector[idx - 1] == 1:
            vvec[idx] = 1
        if (idx + 1) < sz and prefer_vector[idx + 1] == -1:
            vvec[idx] = vvec[idx] + 1
    return vvec


def quality_vector(merged_items):
    quality_vector_list =[item.quality for item in merged_items[1,:]]
    return np.array(quality_vector_list)

class I0:
    def __init__(self, items):
        sz = len(items)
        self.sz = sz
        self.merged_items = merged_items(items, sz)
        self.prefer_vector = prefer_vector(self.merged_items, sz)
        self.vote_vector = vote_vector(self.prefer_vector, sz)
        self.quality_vector=quality_vector(self.merged_items)