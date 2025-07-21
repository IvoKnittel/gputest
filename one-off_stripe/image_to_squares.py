import numpy as np
from extensions2 import extend2
from item import Item

def expand4(n):
    num_tiles = int(np.ceil(n/4))
    return 4*num_tiles+1 , num_tiles

def image2items(image_arr):
    image_items = np.full(image_arr.shape, Item(), dtype=object)
    for i in range(0, image_arr.shape[0]):
        for j in range(0, image_arr.shape[1]):
            image_items[i,j]=Item(int(image_arr[i,j]))
    return image_items


def image_squares(image_arr):
    image_items = image2items(image_arr)
    sz = (image_arr.shape[0],image_arr.shape[1])
    height_expanded, num_tiles_vertical = expand4(sz[0])
    width_expanded, num_tiles_horizontal = expand4(sz[1])

    image_expanded = np.full((height_expanded, width_expanded), Item(), dtype=object)
    image_expanded[0:sz[0],0:sz[1]]=image_items
    image_1x2_items = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_1x2_items[m,n+k] = Item(image_expanded[m, n+k], image_expanded[m+1, n+k])
                image_1x2_items[m+2,n+k] = Item(image_expanded[m+2,n+k], image_expanded[m+3,n+k])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_1x2_items[m+1,n+k] = Item(image_expanded[m+1,n+k], image_expanded[m+2,n+k])
                image_1x2_items[m+3,n+k] = Item(image_expanded[m+3,n+k], image_expanded[m+4,n+k])


    image_2x1_items = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x1_items[m+k,n] = Item(image_expanded[m+k, n], image_expanded[m+k, n+1])
                image_2x1_items[m+k,n+2] = Item(image_expanded[m+k,n+2], image_expanded[m+k,n+3])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x1_items[m+k,n+1] = Item(image_expanded[m+k,n+1], image_expanded[m+k,n+2])
                image_2x1_items[m+k,n+3] = Item(image_expanded[m+k,n+3], image_expanded[m+k,n+4])


    image_2x2_items = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x2_items[m+k,n] = Item(image_1x2_items[m+k,n], image_1x2_items[m+k,n+1])
                image_2x2_items[m+k,n+2] = Item(image_1x2_items[m+k,n+2], image_1x2_items[m+k,n+3])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x2_items[m+k,n+1] = Item(image_1x2_items[m+k,n+1], image_1x2_items[m+k,n+2])
                image_2x2_items[m+k,n+3] = Item(image_1x2_items[m+k,n+3], image_1x2_items[m+k,n+4])

    return image_2x2_items[0:image_arr.shape[0],0:image_arr.shape[1]], image_1x2_items[0:image_arr.shape[0],0:image_arr.shape[1]], image_2x1_items[0:image_arr.shape[0],0:image_arr.shape[1]]


def image_squares2(image_arr):
    image_items = image2items(image_arr)
    sz = (image_arr.shape[0],image_arr.shape[1])
    height_expanded, num_tiles_vertical = expand4(sz[0])
    width_expanded, num_tiles_horizontal = expand4(sz[1])

    image_expanded = np.full((height_expanded, width_expanded), Item(), dtype=object)
    image_expanded[0:sz[0],0:sz[1]]=image_items
    image_all_items = np.full((2*height_expanded, 2*width_expanded), Item(), dtype=object)
    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*m,2*(n+k)] = image_expanded[m, n+k]
                image_all_items[2*(m+2),2*(n+k)] = image_expanded[m+2,n+k]

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*m+1,2*(n+k)] = Item(image_all_items[2*m, 2*(n+k)], image_all_items[2*(m+1), 2*(n+k)])
                image_all_items[2*(m+2)+1,2*(n+k)] = Item(image_all_items[2*(m+2),2*(n+k)], image_all_items[2*(m+3),2*(n+k)])
    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*(m+1)+1,2*(n+k)] = Item(image_all_items[2*m, 2*(n+k)], image_all_items[2*(m+1), 2*(n+k)])
                image_all_items[2*(m+3)+1,2*(n+k)] = Item(image_all_items[2*(m+2),2*(n+k)], image_all_items[2*(m+3),2*(n+k)])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*(m+k),2*n+1] = Item(image_all_items[2*m, 2*(n+k)], image_all_items[2*(m+1), 2*(n+k)])
                image_all_items[2*(m+k),2*(n+2)+1] = Item(image_all_items[2*(m+2),2*(n+k)], image_all_items[2*(m+3),2*(n+k)])
    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*(m+k),2*(n+1)+1] = Item(image_all_items[2*(m+k),2*(n+1)], image_all_items[2*(m+k),2*(n+2)])
                image_all_items[2*(m+k),2*(n+3)+1] = Item(image_all_items[2*(m+k),2*(n+3)], image_all_items[2*(m+k),2*(n+4)])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*(m+k)+1,2*n+1] = Item(image_all_items[2*(m+k)+1,2*n], image_all_items[2*(m+k)+1,2*(n+1)])
                image_all_items[2*(m+k),2*(n+2)+1] = Item(image_all_items[2*(m+k)+1,2*(n+2)], image_all_items[2*(m+k)+1,2*(n+3)])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_all_items[2*(m+k)+1,2*(n+1)+1] = Item(image_all_items[2*(m+k)+1,2*(n+1)], image_all_items[2*(m+k),2*(n+2)])
                image_all_items[2*(m+k)+1,2*(n+3)+1] = Item(image_all_items[2*(m+k)+1,2*(n+3)], image_all_items[2*(m+k),2*(n+4)])

    return image_all_items[0:2*(image_arr.shape[0]),0:2*(image_arr.shape[1])]




def ranks_(vec40, ranks0):
    vec4  = np.sort(vec40)
    ranks = np.sort(ranks0)
    for i in range(1,4):
        if vec4[i]==vec4[i-1]:
            s = np.max([ranks[i],ranks[i-1]])
            r = np.min([ranks[i],ranks[i-1]])
            for j in range(0, 4):
                if ranks[j] > s:
                    ranks[j] = ranks[j] -1
            ranks[i] = r
            ranks[i - 1] = r

    n=len(vec40[vec4<0])
    if n>0:
        ranks = ranks + n - 1
        ranks[vec4<0] = -1
    return ranks.astype(float)


def image_squares_quality(image_squares):
    sz=image_squares.shape
    squares_quality = -np.ones((sz[0],sz[1]), dtype=float)
    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            squares_quality[i,j]=image_squares[i,j].quality

    return squares_quality

def image_squares_ranked0(image_square_quality):
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    sz=(int(image_square_quality.shape[0]/2), int(image_square_quality.shape[1]/2))
    image_square_quality_expand=-np.ones((2*(sz[0]+2),2*(sz[1]+2)), dtype=float)
    image_square_quality_expand[2:2*(sz[0]+1),2:2*(sz[1]+1)]=image_square_quality
    squares_ranked0 = -np.ones((2*sz[0],2*sz[1]), dtype=float)
    for i in range(1, sz[0]+1):
        for j in range(1, sz[0]+1):

            qvec=np.zeros(4, dtype=float)
            q = (i -1, j-1)
            qvec[0] = image_square_quality_expand[2*(q[0]+pos[0][0])+1, 2*(q[1]+pos[0][1])+1]
            qvec[1] = image_square_quality_expand[2*(q[0]+pos[1][0])+1, 2*(q[1]+pos[1][1])+1]
            qvec[2] = image_square_quality_expand[2*(q[0]+pos[2][0])+1, 2*(q[1]+pos[2][1])+1]
            qvec[3] = image_square_quality_expand[2*(q[0]+pos[3][0])+1, 2*(q[1]+pos[3][1])+1]
            ranks = np.argsort(qvec)
            ranks_reduced = ranks_(qvec, ranks)

            # c = 6.0 - np.sum(ranks_reduced)
            ranks_reduced = ranks_reduced # + c/4.0
            q=(2*(i-1),2*(j-1))
            squares_ranked0[q[0]+pos[ranks[0]][0], q[1]+pos[ranks[0]][1]] = ranks_reduced[0]
            squares_ranked0[q[0]+pos[ranks[1]][0], q[1]+pos[ranks[1]][1]] = ranks_reduced[1]
            squares_ranked0[q[0]+pos[ranks[2]][0], q[1]+pos[ranks[2]][1]] = ranks_reduced[2]
            squares_ranked0[q[0]+pos[ranks[3]][0], q[1]+pos[ranks[3]][1]] = ranks_reduced[3]

    return squares_ranked0

def pad_ranked_squares(m):
    r0 = m % 6
    if r0 < 4:
        return 4 - r0

    if r0 == 5:
        return 1


    return 0

def image_squares_ranked(r0):
    # e = (pad_ranked_squares(r.shape[0]),pad_ranked_squares(r.shape[0]))
    sz=r0.shape
    sz_half = (int(sz[0]/2),int(sz[1]/2))
    sz_s = (sz_half[0]+2,sz_half[1]+2)
    r =-np.ones((2*sz_s[0],2*sz_s[1]), dtype=float)
    r[2:sz[0]+2,2:sz[1]+2]=r0
    s = -np.ones((2*sz_s[0],2*sz_s[1]), dtype=float)
    e = [ (0, 0), (0, 1), (1, 0), (1, 1)]
    d_out = [ (e[0][0]-1, e[0][1]-1), (e[1][0]-1, e[1][1]-1), (e[2][0]-1, e[2][1]-1), (e[3][0]-1, e[3][1]-1)]
    d_cmp = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
    d_in  = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i in range(1, s.shape[0]-1):
        for j in range(1, s.shape[1]-1):
            b=(i,j)
            c_out = [(b[0]+d_out[0][0],b[1]+d_out[0][1]), (b[0]+d_out[1][0],b[1]+d_out[1][1]), (b[0]+d_out[2][0],b[1]+d_out[2][1]), (b[0]+d_out[3][0],b[1]+d_out[3][1])]
            c=[(2*c_out[0][0],2*c_out[0][1]), (2*c_out[1][0],2*c_out[1][1]), (2*c_out[2][0],2*c_out[2][1]), (2*c_out[3][0],2*c_out[3][1])]
            f = [(c[0][0] + d_in[0][0], c[0][1] + d_in[0][1]), (c[1][0] + d_in[1][0], c[1][1] + d_in[1][1]),
                 (c[2][0] + d_in[2][0], c[2][1] + d_in[2][1]), (c[3][0] + d_in[3][0], c[3][1] + d_in[3][1])]
            a=np.array([ r[f[0][0], f[0][1] ],r[f[1][0], f[1][1] ], r[f[2][0], f[2][1] ], r[f[3][0], f[3][1]] ])
            g = a[a>-0.5]
            if len(g)>0:
                s[2*i+1, 2*j+1] = np.mean(g)
            else:
                s[2*i+1, 2*j+1] = -1.0
    return s[2:-2,2:-2]

sz_halftile=3

def size_expand1d(n):
    sz_tile=2*sz_halftile
    N = int(np.ceil((n-1)/sz_tile))
    M = int(np.ceil((n + sz_halftile)/sz_tile))
    return max(sz_halftile+sz_tile*N, sz_tile*M),N,M

def size_expand2d(_shape):
    h, Hshifted, H1 = size_expand1d(int(_shape[0]/2))
    w, Wshifted, W1 = size_expand1d(int(_shape[1]/2))
    sz_expand= (h,w)
    return (sz_expand[0],sz_expand[1]), ((H1,W1), (2*Hshifted,2*Hshifted))

def is_free(extension_map,k,l):
    if np.any(extension_map[k:k+2,l:l+2] < 0):
        return False
    return True

occupied = -10.0
blocked = -5.0
emtpy    = 0

def insert_t(extension_tile,idx, more):
    if idx[0] >= extension_tile.shape[0] or idx[1] >= extension_tile.shape[1]:
        return extension_tile

    extension_tile[idx[0]:idx[0] + 2, idx[1]:idx[1] + 2] = occupied
    return extension_tile

def find_best(extension_tile, storage_tile, check):
    best_rank = -1
    best_rank_idx_storage =  (np.nan,np.nan)
    found = False
    r=(0,5)
    if check:
        r=(1,4)
    for k in range(r[0], r[1]):
        for l in range(r[0], r[1]):
            if not is_free(extension_tile, k, l):
                continue

            rank = storage_tile[k, l]
            if rank > best_rank:
                best_rank = rank
                best_rank_idx_storage = (k, l)
                found=True

    return found, best_rank_idx_storage

def find_best0(storage_tile, exclude):
    best_rank = -1
    best_rank_idx_storage =  (np.nan,np.nan)
    found = False
    r=(0,5)
    for k in range(r[0], r[1]):
        for l in range(r[0], r[1]):
            if (k,l) in exclude:
                continue

            if storage_tile[2*k + 1, 2*l + 1] < 0:
                continue

            rank = storage_tile[2*k+1, 2*l+1]
            if rank > best_rank:
                best_rank = rank
                best_rank_idx_storage = (k, l)
                found=True

    return found, best_rank_idx_storage



def find_best1(storage_tile):
    found = False
    exclude=[]
    while not found:
        found, best_rank_idx_storage = find_best0(storage_tile, exclude)
        if not found:
            return False, (np.nan,np.nan)

        found, storage_tile = extend2(storage_tile, best_rank_idx_storage)
        if not found:
            exclude.append(best_rank_idx_storage)

    return found, storage_tile, best_rank_idx_storage



def assign_best(square_extension_tile, square_storage_location_tile, check, images):
    more = None
    found, best_idx_storage = find_best(square_extension_tile, square_storage_location_tile, check)
    if not found:
        return False, False, (np.nan,np.nan), more
    if check:
        found, square_extension_tile, square_storage_location_tile = check_(best_idx_storage, square_extension_tile, square_storage_location_tile, images)

    if not found:
        return False, False, (np.nan,np.nan), more

    if best_idx_storage[0] >= 1 and best_idx_storage[0] <= 3 and best_idx_storage[1] >= 1 and best_idx_storage[1] <= 3:
        return True, True, best_idx_storage, more

    return True, False, best_idx_storage, more

def assign_best0(square_storage_location_tile):
    found, best_idx_storage = find_best0(square_storage_location_tile)
    if not found:
        return False, False, (np.nan,np.nan)

    if best_idx_storage[0] >= 1 and best_idx_storage[0] <= 3 and best_idx_storage[1] >= 1 and best_idx_storage[1] <= 3:
        return True, True, best_idx_storage

    return True, False, (2*best_idx_storage[0]+1,2*best_idx_storage[0]+1)

def insert_best(square_storage_location_tile, upper_left_idx, square_storage_location_map):
    while True:
        found, persistent, best_idx_tile, more = assign_best0(square_storage_location_tile)
        if found:
            if more is None:
                if persistent:
                    best_idx = (best_idx_tile[0] + upper_left_idx[0], best_idx_tile[1] + upper_left_idx[1])
                    square_storage_location_map[best_idx[0]-2:best_idx[0]+2, best_idx[1]-2:best_idx[1]-2] += blocked
                    square_storage_location_map[best_idx[0]-1:best_idx[0]+1, best_idx[1]-1:best_idx[1]-1] += occupied-blocked
                else:
                    sz = square_storage_location_tile.shape
                    b0=[best_idx_tile[0]-2,best_idx_tile[0]+2],[best_idx_tile[1]-2, best_idx_tile[1]-2]
                    b1=[best_idx_tile[0]-1,best_idx_tile[0]+1],[best_idx_tile[0]-1, best_idx_tile[0]-1]
                    blk = [[max(b0[0][0],0), min(b0[0][1],sz[0]-1)],[max(b0[1][0],0), min(b0[1][1],sz[0]-1)]]
                    occ = [[max(b1[0][0],0), min(b1[0][1],sz[0]-1)],[max(b1[1][0],0), min(b1[1][1],sz[0]-1)]]

                    square_storage_location_tile[blk[0][0]:blk[0][1], blk[1][0]:blk[1][1]] += blocked
                    square_storage_location_tile[occ[0][0]:occ[0][1], occ[1][0]:occ[1][1]] += occupied-blocked
        else:
            return False,


# def tile_display_single(square_extension_map, num_tiles_expand_noshift_shift, shift, color_val):
#     sz_tile = 2 * sz_halftile
#     num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][0]
#     num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][1]
#     for I in range(0, num_tiles_expand_row):
#         for J in range(0, num_tiles_expand_col):
#             i=shift[0]*sz_halftile + sz_tile * I
#             j=shift[1]*sz_halftile + sz_tile * J
#             upper_left_idx = (i, j)
#
#             square_extension_map[upper_left_idx[0]:upper_left_idx[0] + sz_tile, upper_left_idx[1]:upper_left_idx[1] + sz_tile] = tile_core_display(square_extension_map[upper_left_idx[0]:upper_left_idx[0] + sz_tile,
#                                              upper_left_idx[1]:upper_left_idx[1] + sz_tile], color_val)
#     return square_extension_map
#
# def tile_display(square_extension_map, num_tiles_expand_noshift_shift, shift1, shift2, color_val):
#     tile_display_single(square_extension_map, num_tiles_expand_noshift_shift, shift1, color_val[0])
#     tile_display_single(square_extension_map, num_tiles_expand_noshift_shift, shift2, color_val[1])
#     return square_extension_map

def image_squares_select_single(square_storage_location_map, num_tiles_expand_noshift_shift, shift):
    sz_tile=2*sz_halftile
    num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][0]
    num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][1]
    for I in range(0, num_tiles_expand_row):
        for J in range(0, num_tiles_expand_col):
            i=shift[0]*sz_halftile + sz_tile * I
            j=shift[1]*sz_halftile + sz_tile * J
            upper_left_idx = (i, j)

            square_storage_location_tile = np.array(square_storage_location_map[upper_left_idx[0]:upper_left_idx[0] + sz_tile -1,
                                                    upper_left_idx[1]:upper_left_idx[1] + sz_tile -1])

            success, square_storage_location_map = insert_best(square_storage_location_tile,
                                                                                   upper_left_idx,
                                                                                   square_storage_location_map)
    return square_storage_location_map


def get_possibles(pos, code, square_extension_tile, square_storage_location_tile):
    possibles =[]
    return possibles

def eval_possibles(pos, code, square_extension_tile, square_storage_location_tile):
    possibles =[]
    return possibles

def image_squares_complete(square_storage_location_map, square_extension_map, num_tiles_expand_noshift_shift, shift):
    sz_tile=2*sz_halftile
    num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][0]
    num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][1]
    for I in range(0, num_tiles_expand_row):
        for J in range(0, num_tiles_expand_col):
            i=shift[0]*sz_halftile + sz_tile * I
            j=shift[1]*sz_halftile + sz_tile * J
            upper_left_idx = (i, j)

            square_extension_tile=square_extension_map[upper_left_idx[0]:upper_left_idx[0] + sz_tile,
            upper_left_idx[1]:upper_left_idx[1] + sz_tile]
            square_storage_location_tile = square_storage_location_map[max(upper_left_idx[0]-1,0):upper_left_idx[0] + sz_tile-1,
                                             max(upper_left_idx[1]-1,0):upper_left_idx[1]-1 + sz_tile]
            posvec, codevec = get_triple_points(square_extension_tile, square_storage_location_tile)
            for k in range(0,len(posvec)):
                possibles = get_possibles(posvec[k], codevec[k], square_extension_tile, square_storage_location_tile)

    return square_extension_map