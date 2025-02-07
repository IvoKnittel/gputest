from items import Item
import numpy as np

def get_preferences(items):

    tmp = []
    for n in range(1, len(items)):
        tmp.append(Item(items[n - 1], items[n]))

    cand_lo = [None] + tmp
    cand_hi = tmp + [None]
    cand = [cand_lo, cand_hi]

    lo_is_better = []
    for n in range(len(items)):
        lo_item = cand_lo[n]
        hi_item = cand_hi[n]

        if lo_item is None:
            lo_is_better.append(False)
        elif hi_item is None:
            lo_is_better.append(True)
        else:
            lo_is_better.append(lo_item.quality > hi_item.quality)

    return np.array([lo_is_better])

class CoupleTripleTemp:
    def __init__(self, lo_is_better):
        sz=len(lo_is_better)
        self.sz = sz
        self.lo_is_better = lo_is_better
        self.hi_is_better = ~lo_is_better
        self.is_couple_begin = np.zeros(sz, dtype=bool)
        self.belongs_couple = np.zeros(sz, dtype=bool)
        self.is_triple_begin = np.zeros(sz, dtype=bool)
        self.belongs_triple = np.zeros(sz, dtype=bool)
        self.belongs_up_chain = np.zeros(sz, dtype=bool)
        self.belongs_down_chain = np.zeros(sz, dtype=bool)
        self.is_single=np.zeros(sz, dtype=bool)
        self.itemlist_idx=np.zeros(sz, dtpype=int)

def couples_chains(lo_is_better):
    a = CoupleTripleTemp(lo_is_better)
    # Initialize previous values
    previous_index = 0
    previous_lo_is_better = a.lo_is_better[0]
    previous_hi_is_better = a.hi_is_better[0]

    # Loop from index 1 to the last element
    for current_index in range(1, a.sz):
        current_lo_is_better = a.lo_is_better[current_index]
        current_hi_is_better = a.hi_is_better[current_index]

        if previous_hi_is_better:
            if current_lo_is_better:
                # We have found a happy couple
                a.is_couple_begin[previous_index] = True
                a.belongs_couple[previous_index] = True
                a.belongs_couple[current_index] = True
            else:
                # We have found a love chain leading upwards (A loves B, but B loves C, but C loves D ...)
                a.belongs_up_chain[previous_index] = True
        else:
            # previous_lo_is_better: the previous item does not like the current one
            if current_lo_is_better:
                # A love chain in the opposite direction
                a.belongs_down_chain[current_index] = True

        # Update previous values
        previous_index = current_index
        previous_lo_is_better = current_lo_is_better
        previous_hi_is_better = current_hi_is_better
    return a

def force_couples(a):
    # merge chain items into couple whenever possible
    for current_idx in range(1, a.sz):
        previous_idx = current_idx -1
        if a.belongs_down_chain[previous_idx] and a.belongs_down_chain[current_idx]:
            # found two chain elements
            a.belongs_down_chain[previous_idx] = False
            a.belongs_down_chain[current_idx] = False
            # force them into a couple
            a.is_couple_begin[previous_idx] = True
            a.belongs_couple[previous_idx] = True
            a.belongs_couple[current_idx] = True

        if a.belongs_up_chain[previous_idx] and a.belongs_up_chain[current_idx]:
            # found two chain elements
            a.belongs_up_chain[previous_idx] = False
            a.belongs_up_chain[current_idx] = False
            # force them into a couple
            a.is_couple_begin[previous_idx] = True
            a.belongs_couple[previous_idx] = True
            a.belongs_couple[current_idx] = True
    

def form_more_couples(a):
    for current_idx in range(0, a.sz-3):
        up = a.belongs_up_chain[current_idx] or a.belongs_down_chain[current_idx] 
        down = a.belongs_up_chain[current_idx+3] or a.belongs_down_chain[current_idx+3]
        if up and down:
           a.belongs_up_chain[current_idx] = False
           a.belongs_up_chain[current_idx+3] = False
           a.belongs_down_chain[current_idx+3] = False
           a.belongs_down_chain[current_idx+3] = False

           a.is_couple_begin[current_idx]=True
           a.is_couple_begin[current_idx+2]=True

           a.belongs_couple[current_idx:current_idx+3]=True
           a.belongs_couple[current_idx:current_idx+3]=True


def form_triples(a):
    for current_idx in range(0, a.sz):
        if a.belongs_up_chain[current_idx]:
            assert current_idx+2 <= (a.sz-1)
            a.is_triple_begin[current_idx]
        if a.belongs_down_chain[current_idx]:
            assert current_idx-2 >= 0
            a.is_triple_begin[current_idx-2]

def eval_triples(a, items):
    triple_start_indices = np.where(a.is_triple_begin)[0]
    merged_item_list=[]
    for idx in range(0, len(triple_start_indices)-1, 2):
        t = triple_start_indices[idx] 
        s = triple_start_indices[idx+1]
        ItemLo1=Item(items[t],items[t+1])
        ItemLo2=Item(items[t+1],items[t+2])

        if ItemLo1.quality > itemLo2.quality: 
            ItemLoSplit = ItemLo1
            ItemLoAll=Item(ItemLoSplit, items[t+2]) 
            single_lo_idx = t+2
            couple_lo_idx = t
        else:
            ItemLoSplit = ItemLo2
            ItemLoAll=Item(ItemLoSplit,items[t])
            single_lo_idx = t
            couple_lo_idx = t+1

        ItemHi1=Item(items[s],items[s+1])
        ItemHi2=Item(items[s+1],items[s+2])

        if ItemHi1.quality > itemHi2.quality:
            ItemHiSplit = ItemHi1
            ItemHiAll=Item(ItemHiSplit,items[s+2])
            single_hi_idx=s+2
            couple_hi_idx=s
        else:
            ItemHiSplit = ItemHi2
            ItemsHiAll=Item(ItemHiSplit,items[s])
            single_hi_idx = s
            couple_hi_idx=s+1

        if ItemLoSplit.quality + ItemsHiAll.quality > ItemLoAll.quality + ItemHiSplit.quality:
            a.is_single[single_lo_idx]=True
            a.is_triple_begin[t]=False
            a.is_couple_begin[couple_lo_idx]=True
            a.belongs_couple[couple_lo_idx]=True
            a.belongs_couple[couple_lo_idx+1]=True            
            a.itemlist_idx[s]=len(merged_item_list)
            merged_item_list.append(ItemHiAll)
        else:
            a.is_single[single_hi_idx]=True
            a.is_triple_begin[s]=False
            a.is_couple_begin[couple_hi_idx]=True
            a.belongs_couple[couple_hi_idx]=True
            a.belongs_couple[couple_hi_idx+1]=True
            a.itemlist_idx[t]=len(merged_item_list)
            merged_item_list.append(ItemLoAll)
    return merged_item_list

def triple_eval_insert(items, a, merged_item_list):
    all_merged=[]
    for idx=range(0,a.sz):
        if a.is_couple_begin[idx]:
            all_merged.append(Item(items[idx],items[idx+1]))
        else if a.is_triple[idx]:
            all_merged.append(merged_item_list[idx])
        else if a.is_single[idx]:
            all_merged.append(items[idx])
    return np.array(all_merged)







