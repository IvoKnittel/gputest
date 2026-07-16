
import numpy as np
import pytest

from image_to_squares import (image_squares,
                              image_squares_quality,
                              image_squares_ranked0,
                              image_squares_ranked,
                              size_expand2d,
                              image_squares_select_single,
                              image_squares_complete)

import matplotlib.pyplot as plt

from test_utils import image_generator, display_superposition
from map_of_squares import (place_square_in_core,
                             map_of_squares_from_quality,
                             get_placement_map)
from closure import do_closure
v=0
h=1


@pytest.fixture
def quality_map_setup():
    """Common setup shared by every test in this file: build one synthetic test image
    via image_generator and run it through the image_squares/quality/ranked pipeline
    down to the half-resolution quality map every test consumes from here.
    """
    binary_image, image_noisy_array = next(image_generator(1, 'noisy_square', seed=None))
    image2x2, _, _ = image_squares(image_noisy_array)
    squares_quality2 = image_squares_quality(image2x2)
    r = image_squares_ranked0(squares_quality2)
    quality_map = image_squares_ranked(r)   # shape (20, 20) for a 40x40 image
    return binary_image, image_noisy_array, image2x2, squares_quality2, r, quality_map


@pytest.fixture
def map_of_squares_setup(quality_map_setup):
    """Common setup shared by test_square_placement, test_graph_spread, and
    test_tiling: the CUDA tile/core layout and the padded map_of_squares built from
    quality_map_setup's quality_map, ready for place_square_in_core calls.
    """
    binary_image, image_noisy_array, image2x2, squares_quality2, r, quality_map = quality_map_setup
    szl = quality_map.shape

    # --- CUDA tile/core layout ---
    # Each tile is 5x5 block-positions; the inner 3x3 is the mutable core.
    # Tiles are non-overlapping in the core: stride = sz_core.
    # The 1-cell border around each core is read-only context (shared with neighbours).
    sz_tile = 5
    sz_core = 3
    sz_border = (sz_tile - sz_core) // 2   # = 1

    num_tiles_v = int(np.ceil(szl[v] / sz_core))
    num_tiles_h = int(np.ceil(szl[h] / sz_core))

    # Pad quality map so every core is fully covered, with a 1-cell read-only border.
    # sz_border = 1 is exactly the reach of the diagonal alert writes in
    # place_square_in_core (best_pos +/- 1), so it's enough to keep those writes
    # in-bounds for cores at the outer edge of the grid too - no extra margin needed.
    padded_rows = sz_border + num_tiles_v * sz_core + sz_border
    padded_cols = sz_border + num_tiles_h * sz_core + sz_border
    quality_padded = -np.ones((padded_rows, padded_cols), dtype=float)
    quality_padded[sz_border:sz_border + szl[v], sz_border:sz_border + szl[h]] = quality_map

    # map_of_squares carries all placement state (quality/state/alerts/link) and is the
    # only thing the algorithm reasons about; the placement_map built at the end is
    # derived from it purely for display.
    map_of_squares = np.empty((padded_rows, padded_cols), dtype=object)
    map_of_squares_from_quality(map_of_squares, quality_padded)

    # shifts select which sublattice of tiles is active in each of the 4 passes, as
    # (row parity, col parity) of the tile index - not a pixel offset. Within one pass,
    # active tiles are 2 tiles (6 core-cells) apart, so their cores and 1-cell borders
    # never touch, which is what makes the direct state writes in place_square_in_core
    # conflict-free.
    shifts = [[0, 0], [0, 1], [1, 0], [1, 1]]

    return binary_image, szl, sz_core, sz_border, num_tiles_v, num_tiles_h, map_of_squares, shifts


def test_image_to_squares(quality_map_setup):
    binary_image, image_noisy_array, image2x2, squares_quality2, r, quality_map = quality_map_setup

    plt.imshow(image_noisy_array, cmap='gray')
    plt.axis('on')
    plt.show()

    plt.imshow(squares_quality2, cmap='gray')
    plt.axis('on')
    plt.show()

    plt.imshow(r, cmap='gray')
    plt.axis('on')
    plt.show()

    plt.imshow(quality_map, cmap='gray')
    plt.axis('on')
    plt.show()

def test_square_placement(map_of_squares_setup):
    binary_image, szl, sz_core, sz_border, num_tiles_v, num_tiles_h, map_of_squares, shifts = map_of_squares_setup

    # --- one CUDA call per tile: place best square in each 3x3 core ---
    # This test only runs a single round over the 4 sublattices - it exercises the
    # placement primitive, not the repeat-until-covered tiling process (see test_tiling).
    for k in range(4):
        for I in range(shifts[k][0], num_tiles_v, 2):
            for J in range(shifts[k][1], num_tiles_h, 2):
                core_origin = (sz_border + I * sz_core, sz_border + J * sz_core)
                place_square_in_core(map_of_squares, core_origin, sz_core)

    placement_map = get_placement_map(map_of_squares)
    # Crop placement result back to original quality map size.
    placement = placement_map[sz_border:sz_border + szl[v], sz_border:sz_border + szl[h]]

    plt.imshow(placement + binary_image[:szl[v], :szl[h]], cmap='gray')
    plt.axis('on')
    plt.show()

# Not specified yet - see the "Open question" note in do_closure's docstring. The
# idea is that get_graphs collects the connected groups of alert_chosen items (by
# graph_id) that form under repeated do_closure calls, and eval_graphs measures how
# far each one spreads spatially (max_graph_extension) and how many nodes it has
# (max_num_nodes), so the "closure keeps graphs bounded to ~10 cells" hypothesis can
# be checked empirically before anyone tries to prove it.
def test_graph_spread(map_of_squares_setup):
    binary_image, szl, sz_core, sz_border, num_tiles_v, num_tiles_h, map_of_squares, shifts = map_of_squares_setup

    for k in range(4):
        for I in range(shifts[k][0], num_tiles_v, 2):
            for J in range(shifts[k][1], num_tiles_h, 2):
                core_origin = (sz_border + I * sz_core, sz_border + J * sz_core)
                place_square_in_core(map_of_squares, core_origin, sz_core)

    graphs = get_graphs(map_of_squares)

    [max_graph_extension, max_num_nodes] = eval_graphs(graphs)


def test_tiling(map_of_squares_setup):
    binary_image, szl, sz_core, sz_border, num_tiles_v, num_tiles_h, map_of_squares, shifts = map_of_squares_setup

    # --- one CUDA call per tile: place best square in each 3x3 core ---
    # Repeat the 4-sublattice round until the whole plane is covered.
    while not is_tiling_complete(map_of_squares):
        for k in range(4):
            for I in range(shifts[k][0], num_tiles_v, 2):
                for J in range(shifts[k][1], num_tiles_h, 2):
                    core_origin = (sz_border + I * sz_core, sz_border + J * sz_core)
                    place_square_in_core(map_of_squares, core_origin, sz_core)

            do_closure(map_of_squares)

    placement_map = get_placement_map(map_of_squares)
    # Crop placement result back to original quality map size.
    placement = placement_map[sz_border:sz_border + szl[v], sz_border:sz_border + szl[h]]

    plt.imshow(placement + binary_image[:szl[v], :szl[h]], cmap='gray')
    plt.axis('on')
    plt.show()