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
v=0
h=1

def get_extension_pairs(extension):
    # site_pair_cross = [0, 0]
    site_pair_parallel0 = [0, 0]
    site_pair_parallel1 = [0, 0]
    D = extension[1]
    dir = extension[1]
    square_idx = np.array(extension[0])
    # site_pair_cross[D] = square_idx[D] + (1 - D) * 3*dir
    # site_pair_cross[1-D] = square_idx[1-D]
    site_pair_parallel0[D] = square_idx[D] + (1 - D) * 2*dir
    site_pair_parallel0[1-D] = square_idx[1-D]-1
    site_pair_parallel1[D] = square_idx[D] + (1 - D) * 2*dir
    site_pair_parallel1[1-D] = square_idx[1-D]+1
    return site_pair_parallel0,site_pair_parallel1


def get_square_extensions(p, near_squares, square_idx, D, dir):
    e = p[D] + 3 * dir
    near_squares[square_idx][1][D] = e
    c = [[-1, -1], [-1, -1]]
    c[0][D] = e
    c[0][1 - D] = p[1 - D] - 1
    hash_neg = 100 * c[0][v] + c[0][h]
    c[1][D] = e
    c[1][1 - D] = p[1 - D] + 1
    hash_pos = 100 * c[0][v] + c[0][h]
    extension_info_neg = (hash_neg, (square_idx, D, -1))
    extension_info_pos = (hash_pos, (square_idx, D, 1))
    return [extension_info_neg, extension_info_pos]


def get_square_extension(p, near_squares, square_idx, D, dir):
    e = p[D] + 3 * dir
    pair_idx = [-1, -1]
    pair_idx[D] = e
    pair_idx[1 - D] = p[1 - D]
    return pair_idx

def get_extension_pair_quality(extension, tile):
    idx1,idx2 = get_extension_pairs(extension)
    quality1 = tile[idx1[v], idx1[h]]
    quality2 = tile[idx2[v], idx2[h]]
    return np.mean(quality1, quality2)

def resolve_conflict(conflict, tile):
    extension_info0 = conflict[0]
    extension0 = extension_info0[1]
    q0 = get_extension_pair_quality(extension0, tile)
    extension_info1 = conflict[1]
    extension1 = extension_info1[1]
    q1 = get_extension_pair_quality(extension1, tile)
    if q0 > q1:
        return (conflict, 0, q0)
    else:
        return (conflict, 1, q1)

def extend(p, near_squares, tile):
    found = False
    L=len(near_squares)
    extensions=[]
    extension_hashes =[]
    conflicts=[]
    for D in (v,h):
        for square_idx in range(0,L):
            diff=p[D] - near_squares[square_idx][0][D]
            if np.abs(diff)==1:
                extension_pair_idx = get_square_extension(p, near_squares, square_idx, D, diff)
                extension_sites=[[-1,-1],[-1,-1]]
                extension_sites[:][D]= extension_pair_idx[D]
                extension_sites[0][1-D] = extension_pair_idx[1-D]-1
                extension_sites[1][1-D] = extension_pair_idx[1-D]+1

                for extension_site in extension_sites:
                    hash= 100 * extension_site[v] + extension_site[h]
                    hashes= np.array(extension_hashes)
                    conflict_indices = np.where(hashes==hash)[0]
                    if conflict_indices: # conflict found
                        conflict_idx=conflict_indices[0]
                        extension = extension_info[1]
                        conflicting_extension=extensions[conflict_idx][1]
                        resolved_conflict=resolve_conflict((extension,conflicting_extension), tile)

                        hash_pair = [extension_info[0], extensions[conflict_idx][0]]
                        hash = hash_pair[resolved_conflict[1]]

                    else:
                        extensions.append(extension_info)
                        extension_hashes.append(extension_info[0])


def extend2(p, near_squares, tile):
    found = False
    L = len(near_squares)
    extensions = []
    extension_hashes = []
    conflicts = []
    for D in (v, h):
        for square_idx in range(0, L):
            diff = p[D] - near_squares[square_idx][0][D]
            if np.abs(diff) == 1:
                extension_pair_idx = get_square_extension(p, near_squares, square_idx, D, diff)
                extension_sites = [[-1, -1], [-1, -1]]
                extension_sites[:][D] = extension_pair_idx[D]
                extension_sites[0][1 - D] = extension_pair_idx[1 - D] - 1
                extension_sites[1][1 - D] = extension_pair_idx[1 - D] + 1

                for extension_site in extension_sites:
                    hash = 100 * extension_site[v] + extension_site[h]
                    hashes = np.array(extension_hashes)
                    conflict_indices = np.where(hashes == hash)[0]
                    if conflict_indices:  # conflict found
                        conflict_idx = conflict_indices[0]

                        extension = (square_idx, extension_pair_idx)
                        conflicting_extension = extensions[conflict_idx][1]
                        resolved_conflict = resolve_conflict((extension, conflicting_extension), tile)

                        hash_pair = [hash, hashes[conflict_idx]]
                        hash = hash_pair[resolved_conflict[1]]


                else:
                    extensions.append(extension_info)
                    extension_hashes.append(extension_info[0])


def crowded(p, tile):
    neighbors = [(p[v] - 2, p[h] - 2), (p[v] - 2, p[h]), (p[v] - 2, p[h] + 2),
                (p[v], p[h] - 2), (p[v], p[h] + 2),
                (p[v] + 2, p[h] - 2), (p[v] + 2, p[h]), (p[v] + 2, p[h] + 2)]
    near_squares = []
    for p in neighbors:
        if tile[p[v], p[h]] < 0:
            squares = [(p[v]-1,p[h]-1),(p[v]-1,p[h]+1),
                       (p[v]+1,p[h]-1),(p[v]+1,p[h]+1)]
            for square_idx in range(0,len(squares)):
                if tile[squares[square_idx][v],squares[square_idx][h]] < 0:
                    found =True
                    for n in near_squares:
                        if squares[square_idx]==n:
                            found=False
                            break
                    if found:
                        near_squares.append([squares[square_idx],-1,-1])

    return len(near_squares) >= 3, near_squares

def get_extend_quality(idx, tile):
    q=-1.0
    return q

def extend2(tile, s):
    found=False
    best_quality=-1.0
    neighbors = [(s[v] - 3, s[h] - 3), (s[v] - 3, s[h] - 1,), (s[v] - 3, s[h] + 1), (s[v] - 3, s[h] + 3),
                (s[v] - 1, s[h] - 3), (s[v] - 1, s[h] + 3),
                (s[v] + 1, s[h] - 3), (s[v] + 1, s[h] + 3),
                (s[v] + 3, s[h] - 3), (s[v] + 3, s[h] - 1), (s[v] + 3, s[h] + 1), (s[v] + 3, s[h] + 3)]
    best_extended_tile=np.array([])
    for p in neighbors:
        if tile[p[v],p[h]]<0:
            continue
        is_crowded, near_squares = crowded(p, tile)
        if not is_crowded:
            continue
        assert(len(near_squares)==3)
        found_now, extended_tile = extend(p,near_squares, tile)
        if found_now:
            quality= get_extend_quality(p, extended_tile)
            if quality >best_quality:
                found=True
                best_quality =quality
                best_extended_tile = extended_tile

    return found, best_extended_tile