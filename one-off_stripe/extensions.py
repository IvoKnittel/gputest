from pytest_check import is_not_none


# def tile_core_display(tile,color):
#     core = tile[1:5,1:5]
#     core[core != -2] = color
#     tile[1:5, 1:5] = core
#     return tile

def is_occupied(neighbors, s, i, j, k, l):
    if i-1<0 or j-1<0:
        return False
    if s[i,j] < 0:
        neighbors.append((i-k,j-l))
    return neighbors

def is_crowded(s, i,j):
    if s[i-2,j] < 0:
        return False

    neighbors = []
    for k in range(j-2,j+2):
        neighbors = is_occupied(neighbors, s, i-2, k,i,j)

    neighbors = is_occupied(neighbors, s, i-1, j-2, i,j)
    neighbors = is_occupied(neighbors, s, i-1, j+1,i,j)
    neighbors = is_occupied(neighbors, s, i, j - 2,i,j)
    neighbors = is_occupied(neighbors, s, i, j + 1,i,j)

    for k in range(j-2,j+2):
        neighbors = is_occupied(neighbors, s, i+1, k,i,j)

    return neighbors


def collect_1x1_crowded(neighbors_vec, s, i, j):
    if i-1 < 0 or j-1 < 0:
        return False, None

    neighbors = is_crowded(s, i - 1, j - 1)
    if len(neighbors)>=3:
        neighbors_vec.append(neighbors)

    return neighbors_vec

def check_inserted2x2(best_idx, square_extension_tile, square_storage_location_tile, images):
    i = best_idx[0]
    j = best_idx[1]
    s = square_storage_location_tile
    t = square_extension_tile
    neighbors1x1=[]
    for k in range(j-1,j+3):
        neighbors1x1 = collect_1x1_crowded(neighbors1x1, s, i - 1, k)

    neighbors1x1 = collect_1x1_crowded(neighbors1x1, s, i, j - 1)
    neighbors1x1 = collect_1x1_crowded(neighbors1x1, s, i, j + 2)
    neighbors1x1 = collect_1x1_crowded(neighbors1x1, s, i + 1, j - 1)
    neighbors1x1 = collect_1x1_crowded(neighbors1x1, s, i + 1, j + 2)

    for k in range(j-1,j+3):
        neighbors1x1 = collect_1x1_crowded(neighbors1x1, s, i + 2, k)

    return neighbors1x1


def get_start_point(neighbors):
   start=0
   start_idx=0
   is_corner=False
   return start, start_idx, is_corner

def get_neighbors_rel(neighbors, start_idx, clockwise):
    neighbors_rel=neighbors
    return neighbors_rel

def neighbors_rel_to_patterns(neighbors_rel):
    pattern=[]
    return pattern

def pattern_absolute(pattern_rel, start_idx, clockwise, pos):
    pattern=[]
    return pattern

import numpy as np
def neighbors1x1_to_patterns(neighbors1x1):
    pos=neighbors1x1[0]
    neighbors= np.sort(neighbors1x1[1])
    #  6 5 4 3
    #  7     2
    #  8     1
    #  9 a b 0
    start, start_idx, is_corner = get_start_point(neighbors)
    clockwise = cyclic_diff(neighbors[start_idx], neighbors[start_idx+1],12) < cyclic_diff(neighbors[start_idx+1], neighbors[start_idx],12)
    neighbors_rel = get_neighbors_rel(neighbors, start_idx, clockwise)
    pattern_rel = neighbors_rel_to_patterns(neighbors_rel)
    pattern = pattern_absolute(pattern_rel, start_idx, clockwise, pos)
    return pattern

def get_extension_patters(neighbors1x1vec):
    extension_patters=[]
    for neighbors1x1 in neighbors1x1vec:
        pattern = neighbors1x1_to_patterns(neighbors1x1)
        if is_not_none(pattern):
            extension_patters.append(pattern)
    return extension_patters

def match_pattern(pattern, previous_pattern):
    success = False
    combined_pattern = []
    return success, combined_pattern

def get_consistent_extension_patters(extension_patters):

    previous_alt_patterns=[]
    for alt_patterns in extension_patters:
        if len(previous_alt_patterns)==0:
            previous_alt_patterns = alt_patterns
        else:
            next_alt_patterns = []
            for previous_pattern in previous_alt_patterns:
                patterns = []
                for pattern in alt_patterns:
                    success, combined_pattern = match_pattern(pattern, previous_pattern)
                    if success:
                        patterns.append(combined_pattern)
                if len(patterns)>0:
                    next_alt_patterns.append(patterns)
            previous_alt_patterns = next_alt_patterns

    consistent_extension_patters = next_alt_patterns
    return consistent_extension_patters


def eval_consistent_extension_patters(patterns):
    if len(patterns)==0:
        return None
    else:
        return patterns[0]

