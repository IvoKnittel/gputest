
import numpy as np

from image_to_squares import (image_squares,
                              image_squares_quality,
                              image_squares_ranked0,
                              image_squares_ranked,
                              size_expand2d,
                              image_squares_select_single,
                              image_squares_complete)

from PIL import Image
import matplotlib.pyplot as plt
v=0
h=1
def display_superposition(a, b):

    # Example arrays (replace with your own)
    A = a.astype(np.uint8)
    B = b.astype(np.uint8)

    # Normalize to [0,1] for alpha
    A_alpha = A.astype(float) / 255.0
    B_alpha = B.astype(float) / 255.0

    # Create RGBA images
    A_rgba = np.zeros((A.shape[v], A.shape[h], 4), dtype=float)
    B_rgba = np.zeros((B.shape[v], B.shape[h], 4), dtype=float)

    # Black for A
    A_rgba[..., 0:3] = 0  # RGB = 0,0,0
    A_rgba[..., 3] = A_alpha  # Alpha

    # Green for B
    B_rgba[..., 0] = 0  # R
    B_rgba[..., 1] = 1  # G
    B_rgba[..., 2] = 0  # B
    B_rgba[..., 3] = B_alpha  # Alpha

    plt.figure(figsize=(6, 6))
    plt.imshow(A_rgba)
    plt.imshow(B_rgba)
    plt.axis('off')
    plt.show()

def test_image_to_squares():
    image_path = 'image2_monochrom.bmp'
    image = Image.open(image_path)
    binary_image = np.ones((40,40))
    binary_image[7:22,7:22]=0
    binary_array = np.array(binary_image).astype(np.uint8)
    binary_array_u8_range = binary_array * 200 + 20  # Low value = 1, High value = 254

    # Generate random noise (-1, 0, 1)
    noise = np.random.choice([-1, 0, 1], size=binary_array_u8_range.shape)

    # Add noise and clip to uint8 range
    image_noisy_array = np.clip(binary_array_u8_range + noise, 0, 255).astype(np.uint8)
    image2x2, _, _ = image_squares(image_noisy_array)
    plt.imshow(image_noisy_array, cmap='gray')
    plt.axis('on')
    plt.show()
    squares_quality2 = image_squares_quality(image2x2)

    plt.imshow(squares_quality2, cmap='gray')
    plt.axis('on')
    plt.show()

    r= image_squares_ranked0(squares_quality2)
    plt.imshow(r, cmap='gray')
    plt.axis('on')
    plt.show()

    square_relative_quality_map = image_squares_ranked(r)

    plt.imshow(square_relative_quality_map, cmap='gray')
    plt.axis('on')
    plt.show()


def place_square_in_core(quality, placement, core_origin, sz_core, check_placement):
    """Simulate one CUDA call: place the best-quality square in the 3x3 core of a tile.

    quality:        full padded quality map (read-only in the border, writable in core)
    placement:      full placement map; 0=empty, 1=placed, -1=blocked
    core_origin:    (row, col) of the top-left of the 3x3 core in padded coordinates
    sz_core:        size of the mutable core (3)
    check_placement: if True, verify compatibility with neighbouring squares before placing
                     (not yet implemented)
    """
    if check_placement:
        raise NotImplementedError("Compatibility check before placement is not yet implemented")

    # Find the highest-quality unoccupied position in the core.
    best_quality = -1.0
    best_pos = None
    for di in range(sz_core):
        for dj in range(sz_core):
            r, c = core_origin[v] + di, core_origin[h] + dj
            if placement[r, c] == 0 and quality[r, c] > best_quality:
                best_quality = quality[r, c]
                best_pos = (r, c)

    if best_pos is not None:
        placement[best_pos[v], best_pos[h]] = 1


def test_square_placement():
    # --- build synthetic test image ---
    binary_image = np.ones((40, 40))
    binary_image[7:22, 7:22] = 0
    binary_array = np.array(binary_image).astype(np.uint8)
    binary_array_u8_range = binary_array * 200 + 20
    noise = np.random.choice([-1, 0, 1], size=binary_array_u8_range.shape)
    image_noisy_array = np.clip(binary_array_u8_range + noise, 0, 255).astype(np.uint8)

    # --- compute per-position 2×2 quality map ---
    image2x2, _, _ = image_squares(image_noisy_array)
    squares_quality2 = image_squares_quality(image2x2)
    r = image_squares_ranked0(squares_quality2)
    quality_map = image_squares_ranked(r)   # shape (20, 20) for a 40x40 image
    szl = quality_map.shape

    # --- CUDA tile/core layout ---
    # Each tile is 5x5 block-positions; the inner 3x3 is the mutable core.
    # Tiles are non-overlapping in the core: stride = sz_core.
    # The 1-cell border around each core is read-only context (shared with neighbours).
    sz_tile = 5
    sz_core = 3
    sz_border = (sz_tile - sz_core) // 2   # = 1

    check_placement = False

    num_tiles_v = int(np.ceil(szl[v] / sz_core))
    num_tiles_h = int(np.ceil(szl[h] / sz_core))

    # Pad quality map so every core is fully covered, with a 1-cell read-only border.
    padded_rows = sz_border + num_tiles_v * sz_core + sz_border
    padded_cols = sz_border + num_tiles_h * sz_core + sz_border
    quality_padded = -np.ones((padded_rows, padded_cols), dtype=float)
    quality_padded[sz_border:sz_border + szl[v], sz_border:sz_border + szl[h]] = quality_map

    # placement_map tracks which block-positions have a square placed (1) or are blocked (-1).
    placement_map = np.zeros((padded_rows, padded_cols), dtype=int)

    # --- one CUDA call per tile: place best square in each 3x3 core ---
    for I in range(num_tiles_v):
        for J in range(num_tiles_h):
            core_origin = (sz_border + I * sz_core, sz_border + J * sz_core)
            place_square_in_core(quality_padded, placement_map, core_origin, sz_core, check_placement)

    # Crop placement result back to original quality map size.
    placement = placement_map[sz_border:sz_border + szl[v], sz_border:sz_border + szl[h]]

    plt.imshow(placement + binary_image[:szl[v], :szl[h]], cmap='gray')
    plt.axis('on')
    plt.show()