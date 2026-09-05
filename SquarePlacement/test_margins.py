"""Exploratory: an n x n free block wrapped in a 1-cell blocked margin, for
n = 2, 3, 4, 5. Map size is (n+4) x (n+4): build_margin_free_map chooses only
its outermost 1-cell ring (index 0 and -1 on every side) - that choice's own
diagonal-blocking side effect (see map_of_squares_from_array) then blocks the
ring just inside it, leaving the n x n interior free. The region anyone
actually cares about - that blocked ring plus the free core, the same shape
this used to build directly as a plain (n+2) x (n+2) field - sits one cell in
from every edge; ROI_MARGIN=1 crops the outermost chosen ring (a pure
construction detail) back off before display, and MARGIN's width=2 accounts
for it plus the blocked ring when colouring/trimming the real-space panel.

n=2 and n=4 (3x3 and 5x5 cores) are expected to fail outright: do_closure's
own initial closure pass raises InvalidTilingError before any placement
happens.

n=3 and n=5 (4x4 and 6x6 cores) run do_closure once, then up to
MAX_RANDOM_PLACEMENTS rounds of place_random_free_cell (pick a random
still-free cell, chase whatever it obligates via forced_closure +
place_squares, reset alert bookkeeping) followed by do_closure(show=True) -
stopping early, before running out of rounds, once is_realmap_cover_complete
confirms the real-space map has no gap left.
"""

import random

import numpy as np

from map_of_squares import StateEnum, InvalidTilingError
from representation import (build_margin_free_map, RealSpaceMargin,
                             display_closure_step, is_realmap_cover_complete)
from closure import (do_closure,
                      forced_closure,
                      place_squares,
                      clear_all_but_state)

MAX_RANDOM_PLACEMENTS = 8
ROI_MARGIN = 1
MARGIN = RealSpaceMargin(width=2)

def place_random_free_cell(m):
    """Place one square at a random still-free cell, chasing whatever it
    obligates (forced_closure + place_squares) - a plain free cell with no
    forces of its own just places itself. Returns False, a no-op, if no free
    cell is left.
    """
    rows, cols = m.shape
    free_positions = [(i, j) for i in range(rows) for j in range(cols)
                      if m[i, j].state == StateEnum.free]
    if not free_positions:
        return False
    pos = random.choice(free_positions)
    forced = forced_closure(m, pos)
    place_squares(m, list(forced))
    clear_all_but_state(m)
    return True


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
                              show_links=True, show_real=True, colormap=colormap,
                              margin=MARGIN, roi_margin=ROI_MARGIN)
    assert raised, "initial closure should be rejected outright now"

def test_margin_free_4x4realmap():
    n=3
    m = build_margin_free_map(n)
    success=False
    do_closure(m, f'margin {n}x{n}: initial closure')
    for k in range(MAX_RANDOM_PLACEMENTS):
        if not place_random_free_cell(m):
            colormap = np.zeros((*m.shape, 3))
            display_closure_step(m, f'margin {n}x{n}: no free cell left',
                                  show_links=True, show_real=True, colormap=colormap,
                                  margin=MARGIN, roi_margin=ROI_MARGIN)
            break
        do_closure(m, f'margin {n}x{n}: random placement {k + 1}', show=True,
                   margin=MARGIN, roi_margin=ROI_MARGIN)
        if is_realmap_cover_complete(m, margin=2):
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
                              show_links=True, show_real=True, colormap=colormap,
                              margin=MARGIN, roi_margin=ROI_MARGIN)
    assert raised, "initial closure should be rejected outright now"

def test_margin_free_6x6realmap():
    n=5
    m = build_margin_free_map(n)
    success=False
    do_closure(m, f'margin {n}x{n}: initial closure')
    for k in range(MAX_RANDOM_PLACEMENTS):
        if not place_random_free_cell(m):
            colormap = np.zeros((*m.shape, 3))
            display_closure_step(m, f'margin {n}x{n}: no free cell left',
                                  show_links=True, show_real=True, colormap=colormap,
                                  margin=MARGIN, roi_margin=ROI_MARGIN)
            break
        do_closure(m, f'margin {n}x{n}: random placement {k + 1}', show=True,
                   margin=MARGIN, roi_margin=ROI_MARGIN)
        if is_realmap_cover_complete(m, margin=2):
            success=True
            break
    assert success==True
