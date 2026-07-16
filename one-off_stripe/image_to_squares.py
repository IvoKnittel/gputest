import numpy as np
from item import Item

# Axis name constants: v = vertical (row), h = horizontal (column).
v = 0
h = 1

# The four corners of a 2×2 block, as (row, col) offsets from the top-left corner.
CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def expand4(n):
    """Return the smallest multiple-of-4 size >= n that fits all overlapping 4-pixel tiles,
    plus the number of tiles. The +1 covers the last overlapping pair at the tile boundary."""
    num_tiles = int(np.ceil(n / 4))
    return 4 * num_tiles + 1, num_tiles


def image2items(image_arr):
    """Wrap every pixel value in an Item object, producing an object array of the same shape."""
    image_items = np.full(image_arr.shape, Item(), dtype=object)
    for i in range(image_arr.shape[v]):
        for j in range(image_arr.shape[h]):
            image_items[i, j] = Item(int(image_arr[i, j]))
    return image_items


def image_squares(image_arr):
    """Build 1×2, 2×1, and 2×2 Item composites for every overlapping pair/quad in the image.

    The image is tiled in non-overlapping 4×4 blocks (expanded to the next multiple of 4).
    Within each tile, all overlapping horizontal pairs (1×2), vertical pairs (2×1), and
    2×2 quads are constructed by combining adjacent Items.

    Returns (image_2x2, image_1x2, image_2x1), each cropped to the original image size.
    """
    image_items = image2items(image_arr)
    rows, cols = image_arr.shape[v], image_arr.shape[h]
    height_expanded, num_tiles_v = expand4(rows)
    width_expanded,  num_tiles_h = expand4(cols)

    # Pad to tiled size; positions beyond the original image remain empty Items.
    img = np.full((height_expanded, width_expanded), Item(), dtype=object)
    img[0:rows, 0:cols] = image_items

    # --- 1×2 vertical pairs (two rows merged into one Item per column position) ---
    pairs_1x2 = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i, 4 * j + k
                # Pairs at even rows within the tile: rows (0,1) and (2,3)
                pairs_1x2[r0,     c0] = Item(img[r0,     c0], img[r0 + 1, c0])
                pairs_1x2[r0 + 2, c0] = Item(img[r0 + 2, c0], img[r0 + 3, c0])
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i, 4 * j + k
                # Pairs at odd rows within the tile: rows (1,2) and (3,4) — cross tile boundary
                pairs_1x2[r0 + 1, c0] = Item(img[r0 + 1, c0], img[r0 + 2, c0])
                pairs_1x2[r0 + 3, c0] = Item(img[r0 + 3, c0], img[r0 + 4, c0])

    # --- 2×1 horizontal pairs (two columns merged into one Item per row position) ---
    pairs_2x1 = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                # Pairs at even columns within the tile: cols (0,1) and (2,3)
                pairs_2x1[r0, c0]     = Item(img[r0, c0],     img[r0, c0 + 1])
                pairs_2x1[r0, c0 + 2] = Item(img[r0, c0 + 2], img[r0, c0 + 3])
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                # Pairs at odd columns: cols (1,2) and (3,4) — cross tile boundary
                pairs_2x1[r0, c0 + 1] = Item(img[r0, c0 + 1], img[r0, c0 + 2])
                pairs_2x1[r0, c0 + 3] = Item(img[r0, c0 + 3], img[r0, c0 + 4])

    # --- 2×2 quads: combine two horizontally adjacent 1×2 pairs ---
    quads_2x2 = np.full((height_expanded, width_expanded), Item(), dtype=object)
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                quads_2x2[r0, c0]     = Item(pairs_1x2[r0, c0],     pairs_1x2[r0, c0 + 1])
                quads_2x2[r0, c0 + 2] = Item(pairs_1x2[r0, c0 + 2], pairs_1x2[r0, c0 + 3])
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                quads_2x2[r0, c0 + 1] = Item(pairs_1x2[r0, c0 + 1], pairs_1x2[r0, c0 + 2])
                quads_2x2[r0, c0 + 3] = Item(pairs_1x2[r0, c0 + 3], pairs_1x2[r0, c0 + 4])

    return (quads_2x2[0:rows, 0:cols],
            pairs_1x2[0:rows, 0:cols],
            pairs_2x1[0:rows, 0:cols])


def image_squares2(image_arr):
    """Build Item composites for every overlapping 1×2, 2×1, and 2×2 neighborhood,
    and store all of them in a single double-resolution grid.

    The output grid is 2× the image size in each dimension. Even-indexed positions (2i, 2j)
    hold single-pixel Items; odd positions (2i+1, 2j) hold vertical pairs, (2i, 2j+1)
    horizontal pairs, and (2i+1, 2j+1) 2×2 quads.

    Returns the combined grid cropped to (2*rows, 2*cols).
    """
    image_items = image2items(image_arr)
    rows, cols = image_arr.shape[v], image_arr.shape[h]
    height_expanded, num_tiles_v = expand4(rows)
    width_expanded,  num_tiles_h = expand4(cols)

    img = np.full((height_expanded, width_expanded), Item(), dtype=object)
    img[0:rows, 0:cols] = image_items

    # Double-resolution grid: even positions = pixels, odd positions = composites.
    grid = np.full((2 * height_expanded, 2 * width_expanded), Item(), dtype=object)

    # Place single pixels at even-even positions.
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i, 4 * j + k
                grid[2 * r0,       2 * c0] = img[r0,     c0]
                grid[2 * (r0 + 2), 2 * c0] = img[r0 + 2, c0]

    # Vertical pairs at odd-row, even-col positions.
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i, 4 * j + k
                grid[2 * r0 + 1,       2 * c0] = Item(grid[2 * r0,       2 * c0], grid[2 * (r0 + 1), 2 * c0])
                grid[2 * (r0 + 2) + 1, 2 * c0] = Item(grid[2 * (r0 + 2), 2 * c0], grid[2 * (r0 + 3), 2 * c0])
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i, 4 * j + k
                grid[2 * (r0 + 1) + 1, 2 * c0] = Item(grid[2 * r0,       2 * c0], grid[2 * (r0 + 1), 2 * c0])
                grid[2 * (r0 + 3) + 1, 2 * c0] = Item(grid[2 * (r0 + 2), 2 * c0], grid[2 * (r0 + 3), 2 * c0])

    # Horizontal pairs at even-row, odd-col positions.
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                grid[2 * r0, 2 * c0 + 1]       = Item(grid[2 * r0, 2 * c0],       grid[2 * r0, 2 * (c0 + 1)])
                grid[2 * r0, 2 * (c0 + 2) + 1] = Item(grid[2 * r0, 2 * (c0 + 2)], grid[2 * r0, 2 * (c0 + 3)])
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                grid[2 * r0, 2 * (c0 + 1) + 1] = Item(grid[2 * r0, 2 * (c0 + 1)], grid[2 * r0, 2 * (c0 + 2)])
                grid[2 * r0, 2 * (c0 + 3) + 1] = Item(grid[2 * r0, 2 * (c0 + 3)], grid[2 * r0, 2 * (c0 + 4)])

    # 2×2 quads at odd-row, odd-col positions.
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                grid[2 * r0 + 1, 2 * c0 + 1]       = Item(grid[2 * r0 + 1, 2 * c0],       grid[2 * r0 + 1, 2 * (c0 + 1)])
                grid[2 * r0,     2 * (c0 + 2) + 1] = Item(grid[2 * r0 + 1, 2 * (c0 + 2)], grid[2 * r0 + 1, 2 * (c0 + 3)])
    for i in range(num_tiles_v):
        for j in range(num_tiles_h):
            for k in range(4):
                r0, c0 = 4 * i + k, 4 * j
                grid[2 * r0 + 1, 2 * (c0 + 1) + 1] = Item(grid[2 * r0 + 1, 2 * (c0 + 1)], grid[2 * r0,     2 * (c0 + 2)])
                grid[2 * r0 + 1, 2 * (c0 + 3) + 1] = Item(grid[2 * r0 + 1, 2 * (c0 + 3)], grid[2 * r0,     2 * (c0 + 4)])

    return grid[0:2 * rows, 0:2 * cols]


def ranks_(quality_vec, argsort_ranks):
    """Adjust argsort ranks so that tied quality values share the same (lower) rank,
    and positions with negative quality (invalid/missing) are marked -1.

    quality_vec:   4-element float array of quality scores (negative = invalid).
    argsort_ranks: argsort of quality_vec (0 = lowest quality).
    Returns adjusted ranks as floats; invalid positions get rank -1.
    """
    sorted_quality = np.sort(quality_vec)
    ranks = np.sort(argsort_ranks)

    # Merge tied adjacent values: both get the lower rank, higher ranks shift down.
    for i in range(1, 4):
        if sorted_quality[i] == sorted_quality[i - 1]:
            high = np.max([ranks[i], ranks[i - 1]])
            low  = np.min([ranks[i], ranks[i - 1]])
            for j in range(4):
                if ranks[j] > high:
                    ranks[j] -= 1
            ranks[i]     = low
            ranks[i - 1] = low

    # Shift valid ranks up by the number of invalid entries, then mark invalids as -1.
    num_invalid = len(quality_vec[sorted_quality < 0])
    if num_invalid > 0:
        ranks = ranks + num_invalid - 1
        ranks[sorted_quality < 0] = -1

    return ranks.astype(float)


def image_squares_quality(image_squares):
    """Extract the scalar .quality value from each Item in the grid into a float array."""
    rows, cols = image_squares.shape[v], image_squares.shape[h]
    quality = -np.ones((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            quality[i, j] = image_squares[i, j].quality
    return quality


def image_squares_ranked0(quality_map):
    """Assign within-group ranks to each position based on its 2×2 quality value.

    quality_map: float array of 2×2 Item quality scores, one per pixel position.
    Non-overlapping 2×2 blocks are ranked independently: 0=worst, 3=best.
    Ties share the lower rank; invalid (negative) positions get rank -1.
    Returns a float array of the same shape as quality_map.
    """
    rows, cols = quality_map.shape[v], quality_map.shape[h]
    ranked = -np.ones((rows, cols), dtype=float)

    for i in range(0, rows - 1, 2):
        for j in range(0, cols - 1, 2):
            quality_values = np.array([quality_map[i + dr, j + dc] for dr, dc in CORNERS])
            argsort = np.argsort(quality_values)
            corner_ranks = ranks_(quality_values, argsort)
            for k in range(4):
                dr, dc = CORNERS[argsort[k]]
                ranked[i + dr, j + dc] = corner_ranks[k]

    return ranked


def pad_ranked_squares(m):
    r0 = m % 6
    if r0 < 4:
        return 4 - r0
    if r0 == 5:
        return 1
    return 0


def image_squares_ranked(ranked_map):
    """Compute a smoothed quality score for each 2×2 block center.

    ranked_map is a full-resolution rank map (from image_squares_ranked0) where each pixel
    holds its rank within its local 2×2 group. This function works at half resolution:
    for each block position (i, j), it looks up the ranks of the 4 surrounding pixels
    (the corners of that block in the original pixel grid) and averages the valid ones.

    The result is a half-resolution map, padded by 2 pixels on each side during computation
    to avoid border effects, then cropped back to the expected output size.
    """
    rows, cols = ranked_map.shape[v], ranked_map.shape[h]
    # half_sz: number of 2×2 blocks along each axis
    half_rows = int(rows / 2)
    half_cols = int(cols / 2)

    # Pad ranked_map by 2 pixels on each side in the padded array r.
    pad_rows = half_rows + 2
    pad_cols = half_cols + 2
    r = -np.ones((2 * pad_rows, 2 * pad_cols), dtype=float)
    r[2:rows + 2, 2:cols + 2] = ranked_map

    # Output: one score per block center, shape (half_rows, half_cols).
    out = -np.ones((half_rows, half_cols), dtype=float)

    # Offsets from block center (i,j) to the 4 surrounding block-top-left positions.
    # Each block's top-left is one step up-left from its center, so offset is corner - 1.
    corner_to_block_topleft = [(dr - 1, dc - 1) for dr, dc in CORNERS]

    for i in range(1, pad_rows - 1):
        for j in range(1, pad_cols - 1):
            # For each corner direction, find the top-left of the neighboring block,
            # then convert to pixel coordinates in r (multiply by 2) and add the corner offset.
            pixel_coords = [
                (2 * (i + d_block[v]) + dr,
                 2 * (j + d_block[h]) + dc)
                for (d_block, (dr, dc)) in zip(corner_to_block_topleft, CORNERS)
            ]

            ranks = np.array([r[pr, pc] for pr, pc in pixel_coords])
            valid_ranks = ranks[ranks > -0.5]

            out[i - 1, j - 1] = np.mean(valid_ranks) if len(valid_ranks) > 0 else -1.0

    return out

sz_halftile=3

def size_expand1d(n):
    sz_tile=2*sz_halftile
    N = int(np.ceil((n-1)/sz_tile))
    M = int(np.ceil((n + sz_halftile)/sz_tile))
    return max(sz_halftile+sz_tile*N, sz_tile*M),N,M

def size_expand2d(_shape):
    """Return the expanded grid size and tile counts for a block-position map of the given shape."""
    expanded_rows, Hshifted, H1 = size_expand1d(_shape[v])
    expanded_cols, Wshifted, W1 = size_expand1d(_shape[h])
    return (expanded_rows, expanded_cols), ((H1, W1), (2 * Hshifted, 2 * Hshifted))

def is_free(extension_map,k,l):
    if np.any(extension_map[k:k+2,l:l+2] < 0):
        return False
    return True

occupied = -10.0
blocked = -5.0
emtpy    = 0

def insert_t(extension_tile,idx, more):
    if idx[v] >= extension_tile.shape[v] or idx[h] >= extension_tile.shape[h]:
        return extension_tile

    extension_tile[idx[v]:idx[v] + 2, idx[h]:idx[h] + 2] = occupied
    return extension_tile

def find_best0(storage_tile, exclude):
    best_rank = -1
    best_rank_idx_storage = (np.nan, np.nan)
    found = False
    # Tile is (sz_tile-1) x (sz_tile-1) = 5x5; valid odd positions are 1,3 → k in range(0, sz_halftile-1)
    for k in range(0, sz_halftile - 1):
        for l in range(0, sz_halftile - 1):
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



def assign_best0(square_storage_location_tile):
    found, best_idx_storage = find_best0(square_storage_location_tile, exclude=set())
    if not found:
        return False, False, (np.nan,np.nan)

    if best_idx_storage[v] >= 1 and best_idx_storage[v] <= 3 and best_idx_storage[h] >= 1 and best_idx_storage[h] <= 3:
        return True, True, best_idx_storage

    return True, False, (2*best_idx_storage[v]+1,2*best_idx_storage[v]+1)

def insert_best(square_storage_location_tile, upper_left_idx, square_storage_location_map):
    while True:
        found, persistent, best_idx_tile, more = assign_best0(square_storage_location_tile)
        if found:
            if more is None:
                if persistent:
                    best_idx = (best_idx_tile[v] + upper_left_idx[v], best_idx_tile[h] + upper_left_idx[h])
                    square_storage_location_map[best_idx[v]-2:best_idx[v]+2, best_idx[h]-2:best_idx[h]-2] += blocked
                    square_storage_location_map[best_idx[v]-1:best_idx[v]+1, best_idx[h]-1:best_idx[h]-1] += occupied-blocked
                else:
                    sz = square_storage_location_tile.shape
                    b0=[best_idx_tile[v]-2,best_idx_tile[v]+2],[best_idx_tile[h]-2, best_idx_tile[h]-2]
                    b1=[best_idx_tile[v]-1,best_idx_tile[v]+1],[best_idx_tile[v]-1, best_idx_tile[v]-1]
                    blk = [[max(b0[0][v],0), min(b0[0][h],sz[v]-1)],[max(b0[1][h],0), min(b0[1][h],sz[h]-1)]]
                    occ = [[max(b1[0][v],0), min(b1[0][h],sz[v]-1)],[max(b1[1][h],0), min(b1[1][h],sz[h]-1)]]

                    square_storage_location_tile[blk[0][v]:blk[0][v], blk[1][h]:blk[1][h]] += blocked
                    square_storage_location_tile[occ[0][v]:occ[0][v], occ[1][h]:occ[1][h]] += occupied-blocked
        else:
            return False,


# def tile_display_single(square_extension_map, num_tiles_expand_noshift_shift, shift, color_val):
#     sz_tile = 2 * sz_halftile
#     num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][v]
#     num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][h]
#     for I in range(0, num_tiles_expand_row):
#         for J in range(0, num_tiles_expand_col):
#             i=shift[v]*sz_halftile + sz_tile * I
#             j=shift[h]*sz_halftile + sz_tile * J
#             upper_left_idx = (i, j)
#
#             square_extension_map[upper_left_idx[v]:upper_left_idx[v] + sz_tile, upper_left_idx[h]:upper_left_idx[h] + sz_tile] = tile_core_display(square_extension_map[upper_left_idx[v]:upper_left_idx[v] + sz_tile,
#                                              upper_left_idx[h]:upper_left_idx[h] + sz_tile], color_val)
#     return square_extension_map
#
# def tile_display(square_extension_map, num_tiles_expand_noshift_shift, shift1, shift2, color_val):
#     tile_display_single(square_extension_map, num_tiles_expand_noshift_shift, shift1, color_val[v])
#     tile_display_single(square_extension_map, num_tiles_expand_noshift_shift, shift2, color_val[h])
#     return square_extension_map

def image_squares_select_single(square_storage_location_map, num_tiles_expand_noshift_shift, shift):
    sz_tile=2*sz_halftile
    num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][v]
    num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][h]
    for I in range(0, num_tiles_expand_row):
        for J in range(0, num_tiles_expand_col):
            i=shift[v]*sz_halftile + sz_tile * I
            j=shift[h]*sz_halftile + sz_tile * J
            upper_left_idx = (i, j)

            square_storage_location_tile = np.array(square_storage_location_map[upper_left_idx[v]:upper_left_idx[v] + sz_tile -1,
                                                    upper_left_idx[h]:upper_left_idx[h] + sz_tile -1])

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
    num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][v]
    num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][h]
    for I in range(0, num_tiles_expand_row):
        for J in range(0, num_tiles_expand_col):
            i=shift[v]*sz_halftile + sz_tile * I
            j=shift[h]*sz_halftile + sz_tile * J
            upper_left_idx = (i, j)

            square_extension_tile=square_extension_map[upper_left_idx[v]:upper_left_idx[v] + sz_tile,
            upper_left_idx[h]:upper_left_idx[h] + sz_tile]
            square_storage_location_tile = square_storage_location_map[max(upper_left_idx[v]-1,0):upper_left_idx[v] + sz_tile-1,
                                             max(upper_left_idx[h]-1,0):upper_left_idx[h]-1 + sz_tile]
            posvec, codevec = get_triple_points(square_extension_tile, square_storage_location_tile)
            for k in range(0,len(posvec)):
                possibles = get_possibles(posvec[k], codevec[k], square_extension_tile, square_storage_location_tile)

    return square_extension_map