import os
import pytest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist

from closure import do_closure, place_squares
from map_of_squares import InvalidTilingError, StateEnum
from representation import (map_of_squares_from_array, display_closure_step,
                             place_blocked_squares, build_map_of_squares)
from image_to_squares import (build_quality_map, image_squares_select_single, tile_counts_2d,
                               insert_tile, sz_halftile, tile_upper_left_indices, best_allowed,
                               core_range_for_tile, insert_best)
import numpy as np

# Set env var DISPLAY_KERNEL_CALLS=1 to pop up the per-kernel-call plots; off by
# default so the assertions below can run headless.
DISPLAY = True # bool(int(os.environ.get("DISPLAY_KERNEL_CALLS", "0")))

sz_tile = 2 * sz_halftile

# Sentinels for the raw-float lattice arrays (m), distinct from a colorcode (0..15):
# FREE is insert_tile's usual background value; BLOCKED/CHOSEN mark cells whose
# state is pre-seeded rather than written by any kernel call.
FREE = -1.0
BLOCKED = -2.0
CHOSEN = -3.0

@pytest.fixture
def quality_map_setup():
    return build_quality_map()


def show_highlighted(m, colorcode, title, cmap_name="tab20", vmin=-1, vmax=15, dim=0.2):
    """Show m under cmap, with every cell whose value isn't colorcode dimmed toward
    black - so the tiles the current kernel call just placed stand out."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(m))
    rgba[m != colorcode, :3] *= dim
    plt.imshow(rgba)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
    plt.title(title)
    plt.show()


def show_states(square_map, title):
    """Show square_map's abstract state grid next to its real-space physical
    footprint (display_closure_step's show_real panel) - one landscape (1, 2)
    figure, abstract map on the left, real map on the right.

    Raises InvalidTilingError if display_closure_step reports a diagonal-chosen
    conflict - real_space_map no longer raises that itself (see its docstring),
    so this is the explicit "if display returns True, throw" checkpoint for
    every caller of show_states."""
    colormap = np.zeros((*square_map.shape, 3))
    error = display_closure_step(square_map, title, show_real=True, colormap=colormap)
    if error:
        raise InvalidTilingError(f"{title}: real_space_map found a diagonal-chosen conflict")


def find_diagonal_chosen_conflicts(square_map):
    """Every pair of StateEnum.chosen cells in square_map that are diagonal
    neighbours of each other - the same conflict real_space_map's own check
    raises InvalidTilingError over. Only checks the two "forward" diagonal
    offsets (down-right, down-left) so each conflicting pair is reported once,
    not twice."""
    rows, cols = square_map.shape
    conflicts = []
    for i in range(rows):
        for j in range(cols):
            if square_map[i, j].state != StateEnum.chosen:
                continue
            for di, dj in ((1, 1), (1, -1)):
                ni, nj = i + di, j + dj
                if (0 <= ni < rows and 0 <= nj < cols
                        and square_map[ni, nj].state == StateEnum.chosen):
                    conflicts.append(((i, j), (ni, nj)))
    return conflicts


def show_states_with_markers(square_map, title, marker_cells, marker_color='red'):
    """Like show_states, but show_real=False - a board with a diagonal-chosen
    conflict would raise from inside real_space_map before anything could be
    drawn otherwise - and with an open circle marker over every (row, col) in
    marker_cells, on top of the usual colorize_with_alerts panel."""
    colormap = np.zeros((*square_map.shape, 3))
    fig, ax = plt.subplots(figsize=(6, 6))
    display_closure_step(square_map, title, show_links=True, show_real=False, ax=ax, colormap=colormap)
    for i, j in marker_cells:
        ax.plot(j, i, 'o', markersize=14, markerfacecolor='none',
                markeredgecolor=marker_color, markeredgewidth=3, zorder=6)
    plt.show()


def seeded_margin_map(sz):
    """Initial lattice state, filled in place (sz is unchanged - no cells added):
    a 1-cell-wide BLOCKED margin around the whole array, FREE interior, and the
    four interior corner cells pre-set to CHOSEN.

    The corners need no computation or do_closure to know: each one sits at a
    map_of_squares "seat" (see closure.place_square_in_seat) purely from the
    margin - e.g. (1, 1)'s own 2x2 block is (0,0)/(0,1)/(1,0)/(1,1), and the first
    three are all margin (BLOCKED), leaving (1, 1) as the single free corner of an
    otherwise-blocked seat, which place_square_in_seat_closed would always fill.
    """
    rows, cols = sz
    m = FREE * np.ones(sz)
    m[0, :] = BLOCKED
    m[-1, :] = BLOCKED
    m[:, 0] = BLOCKED
    m[:, -1] = BLOCKED
    for corner in ((1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)):
        m[corner] = CHOSEN
    return m


def add_blocked_margin(square_map):
    """Set square_map's outer 1-cell ring to StateEnum.blocked, in place - fills
    the existing array, doesn't resize it, same as seeded_margin_map above but for
    a real map_of_squares (see representation.build_margin_free_map, which builds
    this same margin fresh instead of adding it to an existing map)."""
    rows, cols = square_map.shape
    border = [(i, j) for i in range(rows) for j in range(cols)
              if i in (0, rows - 1) or j in (0, cols - 1)]
    place_blocked_squares(square_map, border)
    return square_map


def min_placement_distance(m, colorcode):
    """Minimum center-to-center distance between the disjoint 3x3 cores that this
    kernel call just stamped with colorcode."""
    mask = (m == colorcode)
    labeled, num = label(mask)
    assert num >= 2, f"colorcode={colorcode} produced only {num} placement(s), can't measure a distance"
    centers = center_of_mass(mask, labeled, range(1, num + 1))
    return float(pdist(np.array(centers)).min())


def test_superlattice():
    num_tiles = (10,13)
    sz=(3*num_tiles[0]+2, 3*num_tiles[1]+2)
    m = -np.ones(sz)
    num_tiles_expand_noshift_shift = tile_counts_2d(sz)
    min_distances = []
    for colorcode in range(4):
        m = insert_tile(m, num_tiles_expand_noshift_shift, colorcode)
        min_distances.append(min_placement_distance(m, colorcode))

    if DISPLAY:
        plt.imshow(m, cmap="tab10")
        plt.colorbar()
        plt.title("test_superlattice: insert_tile shift labels")
        plt.show()

    assert min_distances == pytest.approx([min_distances[0]] * len(min_distances))
    assert min_distances[0] == pytest.approx(sz_tile)


def test_supersuperlattice_ordered():
    """16 kernel calls, one per colorcode 0..15, issued in order."""
    num_tiles = (10, 13)
    sz = (3*num_tiles[0]+2, 3*num_tiles[1]+2)
    m = -np.ones(sz)
    num_tiles_expand_noshift_shift = tile_counts_2d(sz)
    min_distances = []
    for colorcode in range(16):
        m = insert_tile(m, num_tiles_expand_noshift_shift, colorcode, super=True)
        min_distances.append(min_placement_distance(m, colorcode))
        if DISPLAY:
            show_highlighted(m, colorcode, f"test_supersuperlattice_ordered: colorcode={colorcode}")

    assert set(range(16)) <= set(np.unique(m).astype(int))
    assert min_distances == pytest.approx([min_distances[0]] * len(min_distances))
    assert min_distances[0] == pytest.approx(2 * sz_tile)


def test_supersuperlattice_random_order():
    """Same 16 kernel calls as test_supersuperlattice_ordered, issued in a random
    permutation - the calls are disjoint-write, so the order must not matter.

    Both m_ordered and m_random start from seeded_margin_map, not a blank FREE
    canvas: a 1-cell BLOCKED margin plus the four corner cells already CHOSEN (no
    computation/do_closure needed for those - see seeded_margin_map). insert_tile
    is called with only_if_equals=FREE so it never clobbers that pre-seeded state,
    only ever filling in cells that are still free."""
    num_tiles = (10, 13)
    sz = (3*num_tiles[0]+2, 3*num_tiles[1]+2)
    num_tiles_expand_noshift_shift = tile_counts_2d(sz)

    m_ordered = seeded_margin_map(sz)
    for colorcode in range(16):
        m_ordered = insert_tile(m_ordered, num_tiles_expand_noshift_shift, colorcode, super=True, only_if_equals=FREE)

    order = np.random.default_rng(seed=0).permutation(16)
    m_random = seeded_margin_map(sz)
    min_distances = []
    for colorcode in order:
        colorcode = int(colorcode)
        m_random = insert_tile(m_random, num_tiles_expand_noshift_shift, colorcode, super=True, only_if_equals=FREE)
        min_distances.append(min_placement_distance(m_random, colorcode))
        if DISPLAY:
            show_highlighted(m_random, colorcode, f"test_supersuperlattice_random_order: colorcode={colorcode}")

    assert np.array_equal(m_ordered, m_random)

    # Every colorcode's tiles sit exactly 2*sz_tile apart from their nearest same-
    # colorcode neighbour, EXCEPT the (at most) two colorcodes whose tile grid
    # includes one of the four seeded CHOSEN corners: only_if_equals=FREE skips
    # writing that one already-occupied cell, so that tile has 8 of its 9 cells
    # instead of 9, shifting its centroid by 1/8 cell along each axis towards the
    # missing corner - giving a nearest-neighbour distance of exactly
    # sqrt((2*sz_tile - 1/8)**2 + (1/8)**2) instead of 2*sz_tile.
    corner_shifted_distance = ((2 * sz_tile - 0.125) ** 2 + 0.125 ** 2) ** 0.5
    for d in min_distances:
        assert d == pytest.approx(2 * sz_tile) or d == pytest.approx(corner_shifted_distance)

def test_square_placement(quality_map_setup):
    binary_image, image_noisy_array, square_map = quality_map_setup

    add_blocked_margin(square_map)
    do_closure(square_map, "test_square_placement")

    num_tiles_expand_noshift_shift = tile_counts_2d(square_map.shape)
    for colorcode in range(4):
        square_map = image_squares_select_single(square_map, num_tiles_expand_noshift_shift, colorcode)


def test_square_placement_random_order_supersuperlattice():
    """Real square placement (do_closure + image_squares_select_single), driven by
    the 16 disjoint supersuperlattice colorcodes in a random order.

    Unlike the toy insert_tile writes in test_supersuperlattice_random_order, real
    placement is order dependent: insert_best picks the single best-quality free cell
    within a core, and do_closure re-runs over the whole map after every placement, so
    which cells are still free (and hence what "best" resolves to) in a not-yet-processed
    core depends on what neighbouring cores already claimed. So this does not - and must
    not - assert the result matches some fixed order.

    A single pass over all 16 colorcodes places at most one square per 3x3 core (that's
    what insert_best does), so one pass alone can never reach full coverage - each core
    still has 8 other cells free afterwards. Repeat the whole 16-colorcode sweep, each in
    its own random order, 4 times over: each further round gets another chance to place a
    square in whatever's still free in a core.

    Even with that, full coverage (is_realmap_cover_complete) is NOT reached after 4
    rounds - two known, unaddressed reasons, noted here rather than worked around:

    1. Margin cells were previously found to stay free forever by accident: find_alerts
       only ever scans the interior (range(1, rows-1) / range(1, cols-1)), so the
       board's outermost ring never gets .alert_chosen and never enters a
       forced_closure chain the way an interior cell can. add_blocked_margin below
       turns that accident into an explicit, intentional border instead (matching
       representation.build_margin_free_map's convention) - real-space coverage still
       excludes it (is_realmap_cover_complete's own margin argument accounts for
       that), but it's no longer an unexplained gap in the interior itself.

    2. Convergence is slow because each kernel call places one square at a time, not the
       group that must be placed together. insert_best's best_allowed pick is a fresh,
       alert-bookkeeping-free cell (no .forces yet), so place_and_chase's forced_closure
       call on it typically returns just {that one cell} - the rest of whatever patch it
       actually belongs to is only discovered later, one do_closure pass at a time, as
       find_alerts/assign_paths gradually builds up .forces elsewhere on the board. Every
       kernel call pays for a full do_closure pass but usually only banks one square.
    """
    _, _, square_map = build_quality_map(seed=42)
    add_blocked_margin(square_map)
    do_closure(square_map, "test_square_placement_random_order_supersuperlattice")

    num_tiles_expand_noshift_shift = tile_counts_2d(square_map.shape)
    rng = np.random.default_rng(seed=0)

    for round_idx in range(4):
        order = rng.permutation(16)
        for colorcode in order:
            colorcode = int(colorcode)
            square_map = image_squares_select_single(
                square_map, num_tiles_expand_noshift_shift, colorcode, super=True)
            if DISPLAY:
                show_states(square_map,
                            f"test_square_placement_random_order_supersuperlattice: "
                            f"round={round_idx} colorcode={colorcode}")

