import numpy as np

from item import Item

def expand4(n):
    num_tiles = int(np.ceil(n/4))
    return 4*num_tiles , num_tiles

def image_squares(image):
    height_expanded, num_tiles_vertical = expand4(image.shape[0])
    width_expanded, num_tiles_horizontal = expand4(image.shape[1])

    image_expanded = np.full((height_expanded, width_expanded), Item(), dtype=object)
    image_expanded[0:image.shape[0],0:image.shape[1]]=np.full(image.shape, Item(1), dtype=object)
    image_1x2_items = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(0, num_tiles_vertical-1):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_1x2_items[m,n+k] = Item(image_expanded[m, n + k], image_expanded[m + 1, n + k])
                image_1x2_items[m+1,n+k] = Item(image_expanded[m+1,n+k], image_expanded[m+2,n+k])

    for i in range(0, num_tiles_vertical-1):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_1x2_items[m+2,n+k] = Item(image_expanded[m+2,n+k], image_expanded[m+3,n+k])
                image_1x2_items[m+3,n+k] = Item(image_expanded[m+3,n+k], image_expanded[m+4,n+k])


    image_2x2_items = np.full((height_expanded+1, width_expanded+1), Item(), dtype=object)
    for i in range(1, num_tiles_vertical):
        for j in range(1, num_tiles_horizontal-1):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x2_items[1+m+k,1+n] = Item(image_1x2_items[m+k,n], image_1x2_items[m+k,n+1])
                image_2x2_items[1+m+k,1+n+1] = Item(image_1x2_items[m+k,n+1], image_1x2_items[m+k,n+2])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal-1):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x2_items[1+m+k,1+n+2] = Item(image_1x2_items[m+k,n+2], image_1x2_items[m+k,n+3])
                image_2x2_items[1+m+k,1+n+3] = Item(image_1x2_items[m+k,n+3], image_1x2_items[m+k,n+4])

    return image_2x2_items

def image_squares_ranked0(image_squares):
    squares_ranked0 = np.zeros((2*(image_squares.shape[0]-1),2*(image_squares.shape[1]-1)), dtype=int)
    for i in range(1, image_squares.shape[0]-1):
        for j in range(1, image_squares.shape[1]-1):

            qvec=np.zeros(4, dtype=float)
            qvec[0] = image_squares[i-1, j-1].quality
            qvec[1] = image_squares[i-1, j].quality
            qvec[2] = image_squares[i, j-1].quality
            qvec[3] = image_squares[i, j].quality
            ranks = np.argsort(qvec)
            q=(2*i,2*j)
            squares_ranked0[q[0], q[1]] = ranks[0]
            squares_ranked0[q[0], q[1] + 1] = ranks[1]
            squares_ranked0[q[0] + 1, q[1]] = ranks[2]
            squares_ranked0[q[0] + 1, q[1] + 1] = ranks[3]

    return squares_ranked0

def pad_ranked_squares(m):
    r0 = m % 6
    if r0 < 4:
        return 4 - r0

    if r0 == 5:
        return 1

    return 0

def image_squares_ranked(r):
    e = (pad_ranked_squares(r.shape[0]),pad_ranked_squares(r.shape[0]))
    s = -np.ones((r.shape[0]+e[0],r.shape[1]+e[1]), dtype=int)
    for i in range(2, int((r.shape[0]-1)/2)):
        for j in range(2, int((r.shape[1]-1)/2)):
            M = np.iinfo(np.int32).max/3 #3 is the max rank value
            s[2 * i, 2 * j] = int(M*np.mean([r[2 * (i - 1) + 1, 2 * (j - 1) + 1], r[2 * (i - 1) + 1, 2 * j], r[2 * i, 2 * (j - 1)], r[2 * i, 2 * j + 1]]))
            s[2 * i, 2 * j + 1] = int(M*np.mean([r[2 * (i - 1) + 1, 2 * j + 1], r[2 * (i - 1) + 1, 2 * (j + 1)], r[2 * i, 2 * j],r[2 * i, 2 * (j + 1) + 1]]))
            s[2 * i + 1, 2 * j] = int(M*np.mean([r[2 * i + 1, 2 * (j - 1) + 1], r[2 * i + 1, 2 * j], r[2 * (i + 1), 2 * (j - 1)],r[2 * (i + 1), 2 * j + 1]]))
            s[2 * i + 1, 2 * j + 1] = int(M*np.mean([r[2 * i + 1, 2 * j], r[2 * i + 1, 2 * (j + 1)], r[2 * (i + 1), 2 * j], r[2 * (i + 1), 2 * (j + 1)]]))
    return s

def size_expand1d(n):
    r= n%6
    N = int(np.ceil((n-1)/6))
    M = int(np.ceil((n + 3) / 6))
    return (max(3+6*N, 6*M),N,M)

def size_expand2d(_shape):
    h, H0, H1 = size_expand1d(_shape[0])
    w, W0, W1 = size_expand1d(_shape[1])
    sz_expand= (h,w)
    return sz_expand, ((H0,W0), (H1,W1))

def is_free(extension_map,k,l):
    if extension_map[k+1,l+1] > -1:
        return False
    if extension_map[k+1,l] > -1:
        return False
    if extension_map[k,l+1] > -1:
        return False
    if extension_map[k,l] > -1:
        return False
    return True

def find_best(extension_map, storage_map):
    best_rank = -1
    best_rank_idx =  (np.nan,np.nan)
    for k in range(0, 5):
        for l in range(0, 5):
            if not is_free(extension_map, k, l):
                continue

            rank = storage_map[k, l]
            if rank > best_rank:
                best_rank = rank
                best_rank_idx = (k, l)
    return best_rank_idx

# def find_best9(storage_map, extension_map):
#     best_ranks = -np.ones((0, 9), dtype=int)
#     best_ranks_idx =  np.empty((0, 9), dtype=(int, int))
#     for k in range(0, 5):
#         for l in range(0, 5):
#             if not is_free(extension_map, k, l):
#                 continue
#
#             rank = storage_map[k, l]
#             for m in range(0, 9):
#                 if rank > best_ranks[m]:
#                     best_ranks[m] = rank
#                     best_ranks_idx[m] = (k, l)
#     return best_ranks_idx

# def is_connected(t,upper_left_idx):
#     if t[upper_left_idx[0]+1,upper_left_idx[1]+1]==1:
#         return True
#     if t[upper_left_idx[0]+1,upper_left_idx[1]+2]==1:
#         return True
#     if t[upper_left_idx[0]+2,upper_left_idx[1]+1]==1:
#         return True
#     if t[upper_left_idx[0]+2,upper_left_idx[1]+1]==2:
#         return True
#     return False

#def is_contested(t,upper_left_idx):
#    return True

# def select_(extension_map,upper_left_idx, best_ranks_idx):
#     sel = []
#     for i in best_ranks_idx:
#         #if is_contested(extension_map,i):
#         #    continue
#         sel.append(i)
#         if is_connected(extension_map, upper_left_idx):
#             break
#
#     return sel

def insert_t(extension_map,idx):
    extension_map[idx[0] + 1, idx[1] + 1] = 1
    extension_map[idx[0] + 1, idx[1]] =1
    extension_map[idx[0], idx[1] + 1] = 1
    extension_map[idx[0], idx[1]]=1


def image_squares_select(square_extension_map, square_storage_location_map):
    sz_expand, num_tiles_expand = size_expand2d(square_storage_location_map.shape)
    square_storage_location_map_expand = -np.ones(sz_expand, dtype=int)
    square_storage_location_map_expand[3:3 + square_storage_location_map.shape[0],
    3:3 + square_storage_location_map.shape[1]] = square_storage_location_map

    for I in range(0, num_tiles_expand[0][0]):
        for J in range(0, num_tiles_expand[0][1]):
            upper_left_idx = (3 + 6 * I, 3 + 6 * J)
            square_extension_tile = square_extension_map[upper_left_idx[0]:upper_left_idx[0] + 6,
                                    upper_left_idx[1]:upper_left_idx[1] + 6]
            square_storage_location_tile = square_storage_location_map_expand[upper_left_idx[0]:upper_left_idx[0] + 5,
                                           upper_left_idx[1]:upper_left_idx[1] + 5]
            best_rank_idx = find_best(square_extension_tile, square_storage_location_tile)
            if np.isnan(best_rank_idx[0]) or np.isnan(best_rank_idx[0]):
                continue
            square_storage_location_map[best_rank_idx[0],square_storage_location_map[1]].selected=True
            insert_t(square_extension_map, best_rank_idx)

    for I in range(0, num_tiles_expand[1][0]):
        for J in range(0, num_tiles_expand[1][1]):
            upper_left_idx = (6 * I, 6 * J)
            square_extension_tile = square_extension_map[upper_left_idx[0]:upper_left_idx[0] + 6,
                                    upper_left_idx[1]:upper_left_idx[1] + 6]
            square_storage_location_tile = square_storage_location_map_expand[upper_left_idx[0]:upper_left_idx[0] + 5,
                                           upper_left_idx[1]:upper_left_idx[1] + 5]
            best_rank_idx = find_best(square_extension_tile, square_storage_location_tile)
            if np.isnan(best_rank_idx[0]) or np.isnan(best_rank_idx[0]):
                continue
            square_storage_location_map[best_rank_idx[0],square_storage_location_map[1]].selected=True
            insert_t(square_extension_map, best_rank_idx)

    return square_extension_map, square_storage_location_map