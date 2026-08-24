"""Exploratory: an n x n free block wrapped in a 1-cell blocked margin, for
n = 2, 3, 4, 5. Map size is (n+2) x (n+2): every border cell is blocked, every
interior cell starts free.

For each n, run the same stages redo_closure runs (find_alerts, remove blocked links,
resolve_cycles_and_centrality), displaying the result - then check whether any
alert_chosen item ended up with a real path_id (non-empty). If so, chase that
patch (forced_closure + place_squares) and display again. Inspecting path_id
has to happen at this point, not after calling redo_closure itself:
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

from map_of_squares import StateEnum, InvalidTilingError
from representation import (build_margin_free_map, RealSpaceMargin,
                             display_closure_step, is_realmap_cover_complete)
from closure import (find_alerts,
                      resolve_cycles_and_centrality,
                      do_closure,
                      forced_closure,
                      get_blocked_links,
                      set_blocked_links,
                      place_squares,
                      reset_alert_bookkeeping)

MAX_RANDOM_PLACEMENTS = 8
MARGIN = RealSpaceMargin(width=1)

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
    set_blocked_links(m, get_blocked_links(m))
    resolve_cycles_and_centrality(m)
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, f'margin {n}x{n}: after resolve_cycles_and_centrality',
                          show_links=True, show_real=True, colormap=colormap, margin=MARGIN)

    # Union every alert_chosen cell's path_id together, rather than collecting
    # them into a set directly - path_id is itself a set now, and a set of sets
    # isn't hashable.
    path_ids = sorted(set().union(*(m[i, j].path_id
                                     for i in range(size) for j in range(size)
                                     if m[i, j].alert_chosen and m[i, j].path_id)))
    print(f'margin {n}x{n}: path_ids found = {path_ids}')

    if path_ids:
        candidates = [(i, j) for i in range(size) for j in range(size)
                      if m[i, j].alert_chosen and path_ids[0] in m[i, j].path_id]
        target = max(candidates, key=lambda p: len(m[p].forces))
        forced = forced_closure(m, target)
        place_squares(m, list(forced))
        reset_alert_bookkeeping(m)
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, f'margin {n}x{n}: after placing patch {path_ids[0]} '
                                 f'(chased from {target})',
                              show_links=True, show_real=True, colormap=colormap, margin=MARGIN)

    for k in range(MAX_RANDOM_PLACEMENTS):
        if not place_random_free_cell(m, size):
            colormap = np.zeros((*m.shape, 3))
            display_closure_step(m, f'margin {n}x{n}: no free cell left',
                                  show_links=True, show_real=True, colormap=colormap, margin=MARGIN)
            break
        do_closure(m, f'margin {n}x{n}: random placement {k + 1}', show=True, margin=MARGIN)
        if is_realmap_cover_complete(m, margin=1):
            break

    return m


def test_margin_free_3x3realmap():
    n=2
    m = build_margin_free_map(n)
    try:
        do_closure(m, f'margin {n}x{n}: initial closure')
        raised = False
    except InvalidTilingError as e:
        raised = True
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, f'margin {n}x{n}: initial closure rejected - {e}',
                              show_links=True, show_real=True, colormap=colormap, margin=MARGIN)
    assert raised, "initial closure should be rejected outright now"

def test_margin_free_4x4realmap():
    n=3
    m = build_margin_free_map(n)
    size = n + 2
    success=False
    do_closure(m, f'margin {n}x{n}: initial closure')
    for k in range(MAX_RANDOM_PLACEMENTS):
        if not place_random_free_cell(m, size):
            colormap = np.zeros((*m.shape, 3))
            display_closure_step(m, f'margin {n}x{n}: no free cell left',
                                  show_links=True, show_real=True, colormap=colormap, margin=MARGIN)
            break
        do_closure(m, f'margin {n}x{n}: random placement {k + 1}', show=True, margin=MARGIN)
        if is_realmap_cover_complete(m, margin=1):
            success=True
            break
    assert success==True



def test_margin_free_5x5realmap():
    n=4
    m = build_margin_free_map(n)
    try:
        do_closure(m, f'margin {n}x{n}: initial closure')
        raised = False
    except InvalidTilingError as e:
        raised = True
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, f'margin {n}x{n}: initial closure rejected - {e}',
                              show_links=True, show_real=True, colormap=colormap, margin=MARGIN)
    assert raised, "initial closure should be rejected outright now"

def test_margin_free_6x6realmap():
    n=5
    m = build_margin_free_map(n)
    size = n + 2
    success=False
    do_closure(m, f'margin {n}x{n}: initial closure')
    for k in range(MAX_RANDOM_PLACEMENTS):
        if not place_random_free_cell(m, size):
            colormap = np.zeros((*m.shape, 3))
            display_closure_step(m, f'margin {n}x{n}: no free cell left',
                                  show_links=True, show_real=True, colormap=colormap, margin=MARGIN)
            break
        do_closure(m, f'margin {n}x{n}: random placement {k + 1}', show=True, margin=MARGIN)
        if is_realmap_cover_complete(m, margin=1):
            success=True
            break
    assert success==True
