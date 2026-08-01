"""Minimal, single-scenario situations - display only, no assertions.

Each test hand-builds one small map_of_squares (via representation.py's builders,
directly setting blocked cells rather than deriving them from placed squares) and
runs it through the full patches-and-conflicts pipeline, displaying the map after
every stage. Contrast with test_complex_graphs.py/test_complex_blocked_links.py,
which assert on specific link/patch/conflict outcomes - the point here is just to
look at what the smallest possible inputs produce, one stage at a time.
"""

from representation import build_map_of_squares, place_blocked_squares, display_closure_step
from closure import (fill_isolated_free_cells,
                      find_alerts,
                      link_patches,
                      remove_blocked_links,
                      check_tiling_invariant)


def test_rectangle():
    """Two vertically-stacked blocked cells, (2, 3) and (3, 3) - the minimal input
    that gets a seat started (see docs/rose_cascades_and_holes/README.md, "Where a
    seat comes from"): each threatens the seat forming immediately to its own side.
    """
    m = build_map_of_squares(7, 7)
    place_blocked_squares(m, [(2, 3), (3, 3)])
    fill_isolated_free_cells(m)
    find_alerts(m)
    link_patches(m)
    remove_blocked_links(m)
    fill_isolated_free_cells(m)
    check_tiling_invariant(m)
    display_closure_step(m, '4: check_tiling_invariant  (invariant held - no state change)', show_links=True)

if __name__ == "__main__":
    test_rectangle()
