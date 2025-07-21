import numpy as np

# + + + + * * *
#  o o o . . .
# +     + * * *
#  o 0 o . . .
# +         * *
#  o o x x . .
# + +   #   * *
#  . . x x . .
# * *       * *
#  . . . . . .
# * * * * * * *
#  . . . . . .
# * * * * * * *
def extend(idx, tile):
    found = False
    extended_tile=np.array(tile)
    return found, extended_tile

def is_crowded(idx, tile):
    found = False
    return found, tile

def get_extend_quality(idx, tile):
    q=-1.0
    return q

def extend2(tile, s):
    best_quality=-1.0
    neighbors = [(s[0] - 3, s[1] - 3,), (s[0] - 3, s[1] - 1,), (s[0] - 3, s[1] + 1,), (s[0] - 3, s[1] + 3),
                (s[0] - 1, s[1] - 3,), (s[0] - 1, s[1] + 3,),
                (s[0] + 1, s[1] - 3,), (s[0] + 1, s[1] + 3,),
                (s[0] + 3, s[1] - 3,), (s[0] + 3, s[1] - 1,), (s[0] + 3, s[1] + 1,), (s[0] + 3, s[1] + 3)]
    for p in neighbors:
        if tile[p[0],p[1]]<0:
            continue
        if not is_crowded(p, tile):
            continue

        found, extended_tile = extend(p, tile)
        if found:
            if get_extend_quality(p, extended_tile)>best_quality:
                return True, extended_tile

    return False, tile