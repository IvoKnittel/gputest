"""Try to construct forbidden shapes to see if it is prevented and how.
"""
import numpy as np

from map_of_squares import InvalidTilingError
from representation import build_map_of_squares, place_blocked_squares, display_closure_step
from closure import do_closure, place_squares
from test_utils import place_and_chase 

def test_pinwheel():
    """Try to build the pinwheel with a single central element. Four chosen squares, arranged in a 90-degree-rotated
    ("pinwheel") pattern around a shared centre: (3,3), (4,5), (6,4), (5,2). 
    """

    m = build_map_of_squares(9, 9)

    place_and_chase(m, (3, 3), "round 1: (3,3) placed")
    place_and_chase(m, (4, 5), "round 2: (4,5) placed")
    place_and_chase(m, (6, 4), "round 3: (6,4) placed")

    try:
        place_and_chase(m, (5, 2), "round 4: (5,2) placed anyway")
        raised = False
    except InvalidTilingError as e:
        raised = True
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, f"round 4: placing_square at (5,2) rejected - {e}",
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
    place_squares(m, [(3, 3), (4, 3), (5, 6), (6, 6)])
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'initial state', show_links=True, show_real=True, colormap=colormap)
    try:
        do_closure(m, '')
        title = 'next state'
    except InvalidTilingError as e:
        title = f'next state (rejected - {e})'
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap)


def test_frozen_area():
    """After round 2, a large stretch of the interior still shows as free, but
    it no longer really is: a single square already pins down everything that
    happens there, so rounds 3-6 just walk through cells whose outcome was
    decided two rounds earlier, not ones that are still open to choice.

    That's harmless for sequential placement - each round's do_closure chase
    resolves a cell before the next one is ever considered, so it never
    matters that the choice was effectively already made. It would matter for
    placing squares in parallel across a superlattice, though: two workers
    each looking at a still-"free" cell in a frozen area could think they have
    an independent choice when only one of them actually does.
    """
    m = build_map_of_squares(11, 11)
    size = 11
    border = [(i, j) for i in range(size) for j in range(size)
              if i in (0, size - 1) or j in (0, size - 1)]
    place_blocked_squares(m, border)
    place_and_chase(m, (4, 4), "round 1: (4,4) placed")
    place_and_chase(m, (5, 4), "round 2: (5,4) placed")
    place_and_chase(m, (6, 7), "round 3: (6,7) placed")
    place_and_chase(m, (5, 9), "round 4: (5,9) placed")
    place_and_chase(m, (2, 9), "round 5: (2,9) placed")
    place_and_chase(m, (8, 9), "round 6: (8,9) placed")

def test_try_3x3_hole1():
    m = build_map_of_squares(12, 12)
    place_and_chase(m, (5, 2), "round 1: (5,2) placed", False)
    place_and_chase(m, (6, 2), "round 2: (6,2) placed", False)
    place_and_chase(m, (5, 8), "round 3: (5,8) placed", False) 
    place_and_chase(m, (6, 8), "round 4: (6,8) placed")   

    place_and_chase(m, (3, 4), "round 5: (3,4) placed") 
    place_and_chase(m, (3, 5), "round 6: (3,5) placed")      

def test_try_3x3_hole2():
    m = build_map_of_squares(12, 12)
    place_and_chase(m, (5, 2), "round 1: (5,2) placed", False)
    place_and_chase(m, (6, 2), "round 2: (6,2) placed", False)
    place_and_chase(m, (5, 8), "round 3: (5,8) placed", False) 
    place_and_chase(m, (6, 8), "round 4: (6,8) placed")   

    place_and_chase(m, (8, 6), "round 5: (8,6) placed") 
    place_and_chase(m, (2, 4), "round 6: (2,4) placed") 
    place_and_chase(m, (8, 5), "round 7: (8,5) placed") 


def test_try_3x3_hole3():
    m = build_map_of_squares(12, 12)
    place_and_chase(m, (3, 5), "round 1: (3,5) placed", False, show_entries_terminals=True)
    place_and_chase(m, (6, 2), "round 2: (6,2) placed", False, show_entries_terminals=True)
    place_and_chase(m, (5, 8), "round 3: (5,8) placed", False, show_entries_terminals=True)
    place_and_chase(m, (9, 5), "round 4: (9,5) placed", False, show_entries_terminals=True)

    place_and_chase(m, (5, 2), "round 5: (5,2) placed", False, show_entries_terminals=True)
    place_and_chase(m, (3, 6), "round 6: (3,6) placed", False, show_entries_terminals=True)
    place_and_chase(m, (9, 4), "round 7: (9,4) placed", show_entries_terminals=True)
    place_and_chase(m, (6, 8), "round 8: (6,8) placed", show_entries_terminals=True)

    #place_and_chase(m, (8, 7), "round 9: (8,7) placed")
    place_and_chase(m, (7, 5), "round 9: (7,5) placed", show_entries_terminals=True)