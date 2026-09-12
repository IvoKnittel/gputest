"""Try to construct forbidden shapes to see if it is prevented and how.
"""
import numpy as np

from map_of_squares import InvalidTilingError, StateEnum
from representation import build_map_of_squares, display_closure_step, place_blocked_squares, real_space_map
from closure import do_closure, place_square, get_seat_positions, place_square_in_seat_closed, clear_all_but_state

def test_pinwheel():
    """Try to build the pinwheel with a single central element. Four chosen squares, arranged in a 90-degree-rotated
    ("pinwheel") pattern around a shared centre: (3,3), (4,5), (6,4), (5,2). 
    """

    m = build_map_of_squares(9, 9)

    do_closure(m, (3, 3), "round 1: (3,3) placed")
    do_closure(m, (4, 5), "round 2: (4,5) placed")
    do_closure(m, (6, 4), "round 3: (6,4) placed")

    try:
        do_closure(m, (5, 2), "round 4: (5,2) placed anyway")
        raised = False
    except InvalidTilingError as e:
        raised = True
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, title=f"round 4: placing_square at (5,2) rejected - {e}",
                              show_links=True, show_real=False, colormap=colormap)
    assert raised, "(5,2) should be rejected outright now"
       
def test_show_other_full_2x2():
    """Show a situation that is not a pinwheel but contains a fully blocked 2x2.

    Documentation only, not an assertion: this shape happens to be a genuine
    impossibility (do_closure now rejects it via check_tiling_invariant), but
    that's not the point being illustrated here - the point is just that a
    fully blocked 2x2 need not be a pinwheel shape. Catch the rejection so the
    test always passes and still shows whichever state resulted.
    """
    m = build_map_of_squares(9, 9)
    for pos in [(3, 3), (4, 3), (5, 6), (6, 6)]:
        place_square(m, pos)
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='initial state', show_links=True, show_real=True, colormap=colormap)
    try:
        do_closure(m)
        title = 'next state'
    except InvalidTilingError as e:
        title = f'next state (rejected - {e})'
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title=title, show_links=True, show_real=True, colormap=colormap)

def test_corner_and_interior_dominoes():
    """(0,0)/(1,0) is a domino sitting right at the board's own corner - a
    cell find_alerts_set_links never scans as the item under consideration
    (its own loop is range(1, rows-1)/range(1, cols-1), interior only), only
    ever read as a ring neighbour of something else. (5,4)/(5,5) is an
    ordinary interior domino. do_closure resolves this cleanly either way
    (verified with show=True too, so the diagonal-chosen-conflict check runs)
    - no known gap demonstrated here, just what the smallest edge-adjacent
    input produces.
    """
    m = build_map_of_squares(9, 9)
    for pos in [(0, 0), (1, 0), (5, 4), (5, 5)]:
        place_square(m, pos)
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='initial state', show_links=True, show_real=True, colormap=colormap)
    try:
        do_closure(m)
        title = 'next state'
    except InvalidTilingError as e:
        title = f'next state (rejected - {e})'
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title=title, show_links=True, show_real=True, colormap=colormap)


def test_try_3x3_hole1():
    m = build_map_of_squares(12, 12)
    do_closure(m, (5, 2), "round 1: (5,2) placed", False)
    do_closure(m, (6, 2), "round 2: (6,2) placed", False)
    do_closure(m, (5, 8), "round 3: (5,8) placed", False) 
    do_closure(m, (6, 8), "round 4: (6,8) placed")   

    do_closure(m, (3, 4), "round 5: (3,4) placed") 
    do_closure(m, (3, 5), "round 6: (3,5) placed")      

def test_try_3x3_hole2():
    m = build_map_of_squares(12, 12)
    do_closure(m, (5, 2), "round 1: (5,2) placed", False)
    do_closure(m, (6, 2), "round 2: (6,2) placed", False)
    do_closure(m, (5, 8), "round 3: (5,8) placed", False) 
    do_closure(m, (6, 8), "round 4: (6,8) placed")   

    do_closure(m, (8, 6), "round 5: (8,6) placed") 
    do_closure(m, (2, 4), "round 6: (2,4) placed") 
    do_closure(m, (8, 5), "round 7: (8,5) placed") 


def test_try_3x3_hole3():
    m = build_map_of_squares(12, 12)
    do_closure(m, (3, 5), "round 1: (3,5) placed", False)
    do_closure(m, (6, 2), "round 2: (6,2) placed", False)
    do_closure(m, (5, 8), "round 3: (5,8) placed", False)
    do_closure(m, (9, 5), "round 4: (9,5) placed", False)

    do_closure(m, (5, 2), "round 5: (5,2) placed", False)
    do_closure(m, (3, 6), "round 6: (3,6) placed", False)
    do_closure(m, (9, 4), "round 7: (9,4) placed")
    do_closure(m, (6, 8), "round 8: (6,8) placed")

    #do_closure(m, (8, 7), "round 9: (8,7) placed")
    do_closure(m, (7, 5), "round 9: (7,5) placed")