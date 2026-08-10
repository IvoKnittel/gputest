"""Try to construct forbidden shapes to see if it is prevented and how.
"""
import numpy as np

from map_of_squares import InvalidTilingError
from representation import build_map_of_squares, display_closure_step
from closure import redo_closure, place_squares
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
    """
    m = build_map_of_squares(9, 9)
    place_squares(m, [(3, 3), (4, 3), (5, 6), (6, 6)])
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'initial state', show_links=True, show_real=True, colormap=colormap)
    redo_closure(m, '')
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'next state', show_links=True, show_real=True, colormap=colormap)


def test_other_full_2x2():

    m = build_map_of_squares(9, 9)
    place_and_chase(m, (3, 3), "round 1: (3,3) placed")
    place_and_chase(m, (4, 3), "round 2: (4,3) placed")
    place_and_chase(m, (5, 6), "round 3: (5,6) placed")
    try:
        place_and_chase(m, (6, 6), "round 4: (6,6) placed anyway")
        raised = False
    except InvalidTilingError as e:
        raised = True
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, f"round 4: place_squares rejected (6,6) - {e}",
                              show_links=True, show_real=False, colormap=colormap)
    assert raised, "(6,6) should be rejected outright now that (5,5) is already chosen"

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
    place_and_chase(m, (3, 5), "round 1: (3,5) placed", False)
    place_and_chase(m, (6, 2), "round 2: (6,2) placed", False)
    place_and_chase(m, (5, 8), "round 3: (5,8) placed", False) 
    place_and_chase(m, (9, 5), "round 4: (9,5) placed", False)   

    place_and_chase(m, (5, 2), "round 5: (5,2) placed", False)   
    place_and_chase(m, (3, 6), "round 6: (3,6) placed", False)
    place_and_chase(m, (9, 4), "round 7: (9,4) placed")   
    place_and_chase(m, (6, 8), "round 8: (6,8) placed") 

    #place_and_chase(m, (8, 7), "round 9: (8,7) placed") 
    place_and_chase(m, (7, 5), "round 9: (7,5) placed") 


def test_free_cell_with_4_direct_blocked_neighbors():
    """ build a free cell with 4 direct blocked neighbors. """
    m = build_map_of_squares(12, 12)
    place_and_chase(m, (2, 2), "round 1: (2,2) placed", True)
    place_and_chase(m, (3, 5), "round 2: (3,5) placed", True)
    place_and_chase(m, (5, 1), "round 3: (5,1) placed", True) 
    place_and_chase(m, (6, 4), "round 4: (6,4) placed", True)   