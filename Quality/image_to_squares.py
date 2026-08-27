import numpy as np
from item import Item
from map_of_squares import SquareItem, StateEnum
from test_utils import place_and_chase
from quality_test_utils import image_generator

# Axis name constants: v = vertical (row), h = horizontal (column).
v = 0
h = 1


def image2items(image_arr):
    """Wrap every pixel value in an Item object, producing an object array of the same shape."""
    image_items = np.full(image_arr.shape, Item(), dtype=object)
    for i in range(image_arr.shape[v]):
        for j in range(image_arr.shape[h]):
            image_items[i, j] = Item(int(image_arr[i, j]))
    return image_items


def image_squares(image_items):
    """Merge every non-overlapping 2x2 block of pixel Items into one Item, halving
    resolution on each axis - image_items.shape must be even on both axes.

    Returns the map of 2x2 Items (one merged Item per block, indexed by block
    position rather than pixel position).
    """
    rows, cols = image_items.shape[v], image_items.shape[h]
    assert rows % 2 == 0 and cols % 2 == 0, f"image_items shape {image_items.shape} must be even"
    item2x2 = np.full((rows // 2, cols // 2), Item(), dtype=object)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            top = Item(image_items[i, j], image_items[i, j + 1])
            bottom = Item(image_items[i + 1, j], image_items[i + 1, j + 1])
            item2x2[i // 2, j // 2] = Item(top, bottom)
    return item2x2


ALLOWED_SIZE_STEP = 3
ALLOWED_SIZE_OFFSET = 2


def check_allowed_size(shape):
    """For now, no padding is supported: a map of 2x2 Items (or the SquareItem map
    built from it) must already have shape (3*N+2, 3*M+2) for integers N,M >= 1 -
    the size image_squares_select_single's tile sweep can cover exactly.
    """
    for n in shape:
        if n < ALLOWED_SIZE_OFFSET + ALLOWED_SIZE_STEP or (n - ALLOWED_SIZE_OFFSET) % ALLOWED_SIZE_STEP != 0:
            raise ValueError(
                f"size {shape} is not of the allowed form (3*N+2, 3*M+2) with N,M >= 1; padding is not supported")


def square_item_map(item2x2):
    """Convert a map of merged 2x2 Items into a same-shape array of SquareItem,
    carrying each Item's .quality across - no ranking, just a direct copy."""
    check_allowed_size(item2x2.shape)
    rows, cols = item2x2.shape
    m = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = SquareItem(quality=item2x2[i, j].quality)
    return m


def build_quality_map(number=1, method='noisy_square', seed=None):
    """Run the image generator and produce a 2D SquareItem array with .quality set
    directly from each non-overlapping 2x2 pixel block of the generated image -
    no ranking, no padding.
    """
    binary_image, image_noisy_array = next(image_generator(number, method, seed=seed))
    image_items = image2items(image_noisy_array)
    item2x2 = image_squares(image_items)
    return binary_image, image_noisy_array, square_item_map(item2x2)


sz_halftile = 3


def tile_counts_for_axis(n):
    """Number of non-overlapping 3-wide cores that exactly cover an axis's valid
    core range (1..n-2), split by shift phase: shift=0 cores start at 1,7,13,...;
    shift=1 cores start at 4,10,16,.... Given the (3*N+2) size promise, the two
    phases together exactly tile every valid core position with no gap and no
    overshoot - no padding needed.
    """
    check_allowed_size((n,))
    N = (n - ALLOWED_SIZE_OFFSET) // ALLOWED_SIZE_STEP
    return (N + 1) // 2, N // 2

def tile_counts_2d(shape):
    """The (shift=0, shift=1) x (row, col) tile-count table image_squares_select_single needs."""
    num0_row, num1_row = tile_counts_for_axis(shape[v])
    num0_col, num1_col = tile_counts_for_axis(shape[h])
    return (num0_row, num0_col), (num1_row, num1_col)

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


def core_range_for_tile(upper_left_idx):
    """The 3x3 mutable core sits 1 cell in from the 5x5 tile's own upper-left
    corner (the 1-cell border is read-only context shared with neighbouring
    tiles), as ((row_start, row_end), (col_start, col_end))."""
    row0, col0 = upper_left_idx[v] + 1, upper_left_idx[h] + 1
    return (row0, row0 + 3), (col0, col0 + 3)


def best_allowed(map_of_squares, core_range):
    """After do_closure has run, find the highest-quality free SquareItem within
    the 3x3 core_range = ((row_start, row_end), (col_start, col_end)) of
    map_of_squares. Returns (found, best_idx) - best_idx is an absolute
    (row, col) index into map_of_squares, or None if the core has no free cell.
    """
    (row_start, row_end), (col_start, col_end) = core_range
    best_idx = None
    best_quality = None
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            if best_quality is None or item.quality > best_quality:
                best_quality = item.quality
                best_idx = (i, j)
    return best_idx is not None, best_idx


def insert_best(square_storage_location_map, upper_left_idx):
    """Place the single highest-quality free SquareItem in the tile's 3x3 core at
    upper_left_idx, then run the closure it triggers."""
    found, best_idx = best_allowed(square_storage_location_map, core_range_for_tile(upper_left_idx))
    if found:
        place_and_chase(square_storage_location_map, best_idx, f"select_single_{best_idx}", show=False)
    return found

def image_squares_select_single(square_storage_location_map, num_tiles_expand_noshift_shift, shift):
    sz_tile=2*sz_halftile
    num_tiles_expand_row = num_tiles_expand_noshift_shift[shift[0]][v]
    num_tiles_expand_col = num_tiles_expand_noshift_shift[shift[1]][h]
    for I in range(0, num_tiles_expand_row):
        for J in range(0, num_tiles_expand_col):
            i=shift[v]*sz_halftile + sz_tile * I
            j=shift[h]*sz_halftile + sz_tile * J
            insert_best(square_storage_location_map, (i, j))
    return square_storage_location_map
