"""Minimal, single-scenario situations - display only, no assertions.

Each test hand-builds one small map_of_squares (via representation.py's builders;
every blocked cell comes from a chosen one - 2026-11-14, Ivo: no blocked cell
whose origin is unexplained) and
runs it through the full patches-and-conflicts pipeline, displaying the map after
every stage. Contrast with test_complex_graphs.py/test_complex_blocked_links.py,
which assert on specific link/patch/conflict outcomes - the point here is just to
look at what the smallest possible inputs produce, one stage at a time.
"""

import numpy as np

from representation import map_of_squares_from_array, display_closure_step
from closure import (do_closure,
                      check_tiling_invariant)
from test_utils import default_display


def test_rectangle():
    """Two vertically-stacked blocked cells, (2, 3) and (3, 3) - the minimal input
    that gets a seat started (see docs/rose_cascades_and_holes/README.md, "Where a
    seat comes from"): each threatens the seat forming immediately to its own side.
    The two are blocked by chosen (1, 4) and (4, 4), to their right, which
    also block (0, 3), (0, 5), (2, 5), (3, 5), (5, 3), (5, 5); everything
    left of column 3 is as it was when the pair was set blocked directly.
    """
    grid = np.zeros((7, 7), dtype=int)
    for pos in [(1, 4), (4, 4)]:
        grid[pos] = 1
    m = map_of_squares_from_array(grid)
    do_closure(m, display=default_display())
    check_tiling_invariant(m)
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='4: check_tiling_invariant  (invariant held - no state change)', show_links=True, colormap=colormap)

if __name__ == "__main__":
    test_rectangle()
