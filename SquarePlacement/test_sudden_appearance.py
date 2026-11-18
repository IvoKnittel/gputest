import numpy as np

from map_of_squares import StateEnum
from representation import build_map_of_squares, build_margin_free_map, map_of_squares_from_array, display_closure_step
from closure import do_closure
from test_utils import default_display

def test_seat_from_two_alert_blocked():
    """(7, 5) used to end up StateEnum.chosen with an empty .forced_by1
    throughout - the .forces1/.forced_by1 link analysis (find_alerts_set_links/
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
    .forced_by1 (hence a real path_id) the moment it's discovered, not only
    once place_square_in_seat_closed's raw scan happens to pick it up. See
    get_seat_positions's "Known gap" docstring (closure.py) for the general
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
    do_closure(m, title='initial', display=default_display(show=True))
    assert m[7,5].forced_by1 == {(5, 4)}

    do_closure(m, (5, 4), "round 1: (5,4) placed", display=default_display())
    assert m[7,5].state == StateEnum.chosen

def test_frozen_area():
    """Same phenomenon as test_seat_from_two_alert_blocked above, from a third
    placement instead of a second - and fixed the same way. (9, 8) has
    forces1={(10, 6), (10, 8)} (something it would force onward) before round 3;
    by the end of round 2, find_secondary_links has already linked both
    (7, 8) and (7, 9) - the two cells round 3 is about to place - to (9, 8)
    directly, so forced_closure(m, (7, 8)) now reaches (9, 8) too, instead of
    stopping short and leaving it for place_square_in_seat_closed's raw scan
    to pick up once round 3's placement happens to complete its seat.
    """

    # the border ring is blocked by an outer ring of chosen cells
    # (build_margin_free_map), not set blocked directly (2026-11-14, Ivo:
    # "blocked cells whose origin is unexplained") - one ring further out,
    # so every coordinate is the old one + 1
    m = build_margin_free_map(9)
    do_closure(m, (5, 5), "round 1: (5,5) placed", display=default_display())
    do_closure(m, (6, 5), "round 2: (6,5) placed", display=default_display(show_all=True))
    assert m[9,8].forced_by1 == {(7, 8), (7, 9)}
    do_closure(m, (7, 8), "round 3: (7,8) placed", display=default_display(show_all=True))
    assert m[9,8].state == StateEnum.chosen
