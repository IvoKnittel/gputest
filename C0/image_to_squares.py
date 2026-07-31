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


def image_squares_quality(image_squares):
    """Extract the scalar .quality value from each Item in the grid into a float array."""
    rows, cols = image_squares.shape[v], image_squares.shape[h]
    quality = -np.ones((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            quality[i, j] = image_squares[i, j].quality
    return quality