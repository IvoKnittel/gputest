import numpy as np

from map_of_squares import StateEnum
from representation import build_map_of_squares, map_of_squares_from_array, place_blocked_squares, display_closure_step
from test_utils  import  place_and_chase
from closure import do_closure, forced_closure

def test_seat_from_two_alert_blocked():
    """(7, 5) used to end up StateEnum.chosen with an empty .forced_by
    throughout - the .forces/.forced_by link analysis (find_alerts_set_links/
    assign_paths) never recorded it as forced by anything, before or after,
    and it was chosen purely by place_square_in_seat_closed's raw state scan
    once placing (5, 4) happened to complete that seat. Fixed by
    find_secondary_links (closure.py): (5, 4) is a free diagonal neighbour of
    alert_blocked (6, 3), and blocking (6, 3) already promises corner (7, 3).
    find_secondary_links simulates (5, 4) and (7, 3) both chosen together -
    which blocks (6, 5) and (6, 4) respectively - and finds that this,
    combined with the real pre-existing block at (7, 4), completes a fresh
    seat at (7, 5): (5, 4) gets linked to it directly, so
    forced_closure(m, (5, 4)) now reaches (7, 5) too, not just (7, 3).

    This also means get_blocked_links's contradiction detection (see
    test_complex_blocked_links.test_get_and_set_blocked_links_marks_blocked_tmp)
    can now see this cell where it couldn't before: (7, 5) has a real
    .forced_by (hence a real path_id) the moment it's discovered, not only
    once place_square_in_seat_closed's raw scan happens to pick it up. See
    place_square_in_seat's "Known gap" docstring (closure.py) for the general
    shape of what's still unaddressed - two independently-forced seats that
    turn out to be diagonal neighbours of each other - which this fix
    narrows but does not claim to close in general.
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
    assert m[7,5].forced_by == {(5, 4)}
    assert forced_closure(m, (5, 4)) == {(5, 4), (7, 3), (7, 5)}
    place_and_chase(m, (5, 4), "round 1: (5,4) placed")
    assert m[7,5].state == StateEnum.chosen

def test_frozen_area():
    """Same phenomenon as test_seat_from_two_alert_blocked above, from a third
    placement instead of a second - and fixed the same way. (8, 7) has
    forces={(9, 5), (9, 7)} (something it would force onward) before round 3;
    by the end of round 2, find_secondary_links has already linked both
    (6, 7) and (6, 8) - the two cells round 3 is about to place - to (8, 7)
    directly, so forced_closure(m, (6, 7)) now reaches (8, 7) too, instead of
    stopping short and leaving it for place_square_in_seat_closed's raw scan
    to pick up once round 3's placement happens to complete its seat.
    """
    m = build_map_of_squares(11, 11)
    size = 11
    border = [(i, j) for i in range(size) for j in range(size)
              if i in (0, size - 1) or j in (0, size - 1)]
    place_blocked_squares(m, border)
    place_and_chase(m, (4, 4), "round 1: (4,4) placed")
    place_and_chase(m, (5, 4), "round 2: (5,4) placed")
    assert m[8,7].forced_by == {(6, 7), (6, 8)}
    assert (8, 7) in forced_closure(m, (6, 7))
    place_and_chase(m, (6, 7), "round 3: (6,7) placed")
    assert m[8,7].state == StateEnum.chosen
