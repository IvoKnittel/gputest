import numpy as np

from map_of_squares import StateEnum
from representation import build_map_of_squares, map_of_squares_from_array, place_blocked_squares, display_closure_step
from test_utils  import  place_and_chase
from closure import do_closure

def test_seat_from_two_alert_blocked():
    """(7, 5) ends up StateEnum.chosen with an empty .forced_by throughout -
    the .forces/.forced_by link analysis (find_alerts/assign_paths) never
    records it as forced by anything, before or after. It's chosen purely by
    place_square_in_seat_closed's raw state scan (three corners blocked, one
    free) once placing (5, 4) happens to complete that seat -
    forced_closure(m, (5, 4)) itself only ever reaches {(5, 4), (7, 3)}, not
    (7, 5) - the seat-scan mechanism is a second, completely separate route to
    StateEnum.chosen that the link analysis never sees.

    This is exactly why get_blocked_links's contradiction detection (see
    test_get_and_set_blocked_links_marks_blocked_tmp below) can't catch two
    seat-scan-derived choices contradicting each other: it only ever checks
    path_id membership, and a cell chosen this way never gets a path_id (or
    any .forced_by) assigned to it at all - there's nothing for
    get_blocked_links to inspect. See place_square_in_seat's "Known gap"
    docstring (closure.py) for the concrete failure this enables: two such
    "escaped" choices that turn out to be diagonal neighbours of each other,
    caught nowhere (Quality/test_image_to_squares.py's
    test_selfblocking_seats is the minimal repro of that).
    """
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    do_closure(m, 'initial', show=True)
    assert m[7,5].forced_by == set()
    place_and_chase(m, (5, 4), "round 1: (5,4) placed")
    assert m[7,5].state == StateEnum.chosen

def test_frozen_area():
    """Same phenomenon as test_seat_from_two_alert_blocked above, from a third
    placement instead of a second: (8, 7) already has forces={(9, 5), (9, 7)}
    (something it would force onward) but forced_by=set() (nothing forces it)
    before round 3, and forced_closure(m, (6, 7)) - what round 3 actually
    places - never reaches (8, 7) either. It still ends up StateEnum.chosen,
    with forced_by still empty afterward, purely via
    place_square_in_seat_closed's raw scan once round 3's placement happens to
    complete its seat.
    """
    m = build_map_of_squares(11, 11)
    size = 11
    border = [(i, j) for i in range(size) for j in range(size)
              if i in (0, size - 1) or j in (0, size - 1)]
    place_blocked_squares(m, border)
    place_and_chase(m, (4, 4), "round 1: (4,4) placed")
    place_and_chase(m, (5, 4), "round 2: (5,4) placed")
    assert m[8,7].forced_by ==set()
    place_and_chase(m, (6, 7), "round 3: (6,7) placed")
    assert m[8,7].state == StateEnum.chosen
