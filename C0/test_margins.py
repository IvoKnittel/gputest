"""Exploratory: an n x n free block wrapped in a 1-cell blocked margin, for
n = 2, 3, 4, 5. Map size is (n+2) x (n+2): every border cell is blocked, every
interior cell starts free.

For each n, run the same stages redo_closure runs (find_alerts, link_patches,
resolve_cycles_and_centrality), displaying the result - then check whether any
alert_chosen item ended up with a real patch_id (not -1). If so, chase that
patch (forced_closure + place_squares) and display again. Inspecting patch_id
has to happen at this point, not after calling redo_closure itself:
remove_blocked_links (redo_closure's last step) resets .patch_id back to -1 for
everything except cells it has itself turned StateEnum.blocked, so the
"is there a patch id" question is only answerable before that reset runs -
exactly where redo_closure's own optional display fires. remove_blocked_links
still runs at the end of each test, to leave the map in the same state
redo_closure itself would.

No assertions - display-only, matching test_simple_situations.py/
test_rose_cascades_and_holes.py's style: the point here is to look at what
these specific inputs produce, not to pin down expected behaviour with
asserts.

After that first chase, each test keeps going with up to 4 more rounds of
random placement: pick a random still-free cell, chase whatever it obligates
(forced_closure + place_squares), redo closure, and display - stopping early,
before 4 rounds, the moment no free cell is left to place.
"""

import random

import numpy as np

from map_of_squares import StateEnum
from representation import (build_margin_free_map,
                             display_closure_step, is_realmap_cover_complete)
from closure import (find_alerts,
                      link_patches,
                      resolve_cycles_and_centrality,
                      redo_closure,
                      forced_closure,
                      place_squares,
                      reset_alert_bookkeeping)

MAX_RANDOM_PLACEMENTS = 8

def place_random_free_cell(m, size):
    """Place one square at a random still-free cell, chasing whatever it
    obligates (forced_closure + place_squares) - a plain free cell with no
    forces of its own just places itself. Returns False, a no-op, if no free
    cell is left.
    """
    free_positions = [(i, j) for i in range(size) for j in range(size)
                      if m[i, j].state == StateEnum.free]
    if not free_positions:
        return False
    pos = random.choice(free_positions)
    forced = forced_closure(m, pos)
    place_squares(m, list(forced))
    reset_alert_bookkeeping(m)
    return True


def run_margin_free_case(n):
    m = build_margin_free_map(n)
    size = n + 2

    find_alerts(m)
    link_patches(m)
    resolve_cycles_and_centrality(m)
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, f'margin {n}x{n}: after resolve_cycles_and_centrality',
                          show_links=True, show_pivots=True, show_real=True, colormap=colormap)

    patch_ids = sorted({m[i, j].patch_id
                         for i in range(size) for j in range(size)
                         if m[i, j].alert_chosen and m[i, j].patch_id != -1})
    print(f'margin {n}x{n}: patch_ids found = {patch_ids}')

    if patch_ids:
        candidates = [(i, j) for i in range(size) for j in range(size)
                      if m[i, j].alert_chosen and m[i, j].patch_id == patch_ids[0]]
        target = max(candidates, key=lambda p: len(m[p].forces))
        forced = forced_closure(m, target)
        place_squares(m, list(forced))
        reset_alert_bookkeeping(m)
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, f'margin {n}x{n}: after placing patch {patch_ids[0]} '
                                 f'(chased from {target})',
                              show_links=True, show_pivots=True, show_real=True, colormap=colormap)

    for k in range(MAX_RANDOM_PLACEMENTS):
        if not place_random_free_cell(m, size):
            break
        redo_closure(m, f'margin {n}x{n}: random placement {k + 1}', show=True)
        if is_realmap_cover_complete(m):
            break

    return m


def test_margin_free_2x2():
    run_margin_free_case(2)


def test_margin_free_3x3():
    run_margin_free_case(3)


def test_margin_free_4x4():
    run_margin_free_case(4)


def test_margin_free_5x5():
    run_margin_free_case(5)
