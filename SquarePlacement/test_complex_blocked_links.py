import numpy as np

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, place_blocked_squares, display_closure_step
from closure import clear_all_but_state, find_alerts_set_links, assign_paths, dissolve_blocked_paths, get_blocked_links, place_square_in_seat_closed, check_tiling_invariant, do_closure

def test_two_forced_cells_block_each_other():
    """A 12x12 map: a 1-cell blocked margin wrapped around the 10x10 area that used
    to be the whole map, so the forces/path_id behaviour under test comes from the
    shape itself, not from free cells touching the array's own edge.

    Runs the current standard round twice, by hand rather than via do_closure,
    since this test wants to check each round's own get_blocked_links result
    and only display the second round if it actually changed anything -
    do_closure's show=True form always displays, with no such check.
    display_closure_step ignores whatever ax it's given whenever show_real=True
    (it "draws its own two-panel figure regardless of ax", see its own
    docstring), so passing it a hand-built ax here would only leave that ax's
    figure empty while the real drawing happened in a figure of its own -
    ax=None (the default) is the only correct choice once show_real=True.

    Same grid, same pipeline, as test_get_and_set_blocked_links_marks_blocked_tmp
    - that test owns the specific get_blocked_links/dissolve_blocked_paths
    assertions (exactly which ids/positions get flagged, why, and which one
    of them has no local corroboration at all). This test only checks the
    board that results, and the two-round display/redo behaviour around it.
    """

    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    size = m.shape[0]
    border = [(i, j) for i in range(size) for j in range(size)
              if i in (0, size - 1) or j in (0, size - 1)]
    place_blocked_squares(m, border)
    find_alerts_set_links(m)
    # Only the two hand-placed dominoes are chosen yet, and .forces/.forced_by
    # (find_alerts_set_links's own output) are already real - path_id
    # (assign_paths) and the blocked-links procedure (get_blocked_links/
    # dissolve_blocked_paths) haven't run yet, so this is the alert/link
    # structure in its rawest, pre-path form.
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'initial alerts/links, before blocked-links',
                          show_links=True, show_real=True, colormap=colormap)

    assign_paths(m)
    dissolve_blocked_paths(m, get_blocked_links(m))

    place_square_in_seat_closed(m)
    # The two hand-placed dominoes survive, plus five seats that newly-
    # permanent blocking completed.
    assert {(i, j) for i in range(size) for j in range(size)
            if m[i, j].state == StateEnum.chosen} == {
        (4, 7), (5, 7), (8, 3), (8, 4),
        (1, 1), (1, 10), (6, 5), (10, 1), (10, 10)}
    check_tiling_invariant(m)

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'after', show_links=True, show_real=True, colormap=colormap)

    state_before_redo = [[m[i, j].state for j in range(size)] for i in range(size)]
    clear_all_but_state(m)

    find_alerts_set_links(m)
    assign_paths(m)
    dissolve_blocked_paths(m, get_blocked_links(m))
    place_square_in_seat_closed(m)

    state_after_redo = [[m[i, j].state for j in range(size)] for i in range(size)]
    changed = state_after_redo != state_before_redo
    assert not changed
    check_tiling_invariant(m)

    if changed:
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, 'closure redone', show_links=True, show_real=True, colormap=colormap)

def test_get_and_set_blocked_links_marks_blocked_tmp():
    """Same shape as test_two_forced_cells_block_each_other, but stopping
    right after get_blocked_links/dissolve_blocked_paths to look at what
    they actually do. Both need path_id to already be real (run after
    assign_paths, do_closure's own order) - calling get_blocked_links straight
    after find_alerts_set_links, without assign_paths in between, leaves every path_id
    empty and returns an empty set too, not because nothing is wrong but
    because there was nothing yet for it to check.

    get_blocked_links(m) returns {26, 33, 62, 64, 90, 117} - six
    self-contradicting ids, found across the whole board, not just the two
    hand-placed squares' immediate neighbourhood. Each is a cell's own
    unique_id, shared with one of its own free diagonal neighbours: e.g. 64
    is (5, 4)'s own id - one of its diagonal neighbours ((4, 5), (6, 5), or
    (6, 3), all three) carries 64 too, so choosing (5, 4) would block a
    fellow member of its own path. dissolve_blocked_paths then blocks each
    id's own origin cell (the one whose own unique_id it is) directly, and
    does nothing else:
    (2, 2), (2, 9), (5, 2), (5, 4), (7, 6), (9, 9) all end up
    StateEnum.blocked, full stop - none of their own .forces/.forced_by is
    cleared (e.g. (2, 2) still carries forces={(1, 2), (4, 1), (2, 1), (1, 4)}
    afterward, now pointing nowhere meaningful), nothing retracts them from
    any other cell's .forces/.forced_by either, and no cell's path_id
    anywhere on the board is stripped of these six ids (38 other cells still
    carry one, unchanged). All of that is left for do_closure's own second
    round to sort out instead: clear_all_but_state wipes every cell's
    .forces/.forced_by/.path_id unconditionally, and the fresh
    find_alerts_set_links/assign_paths pass that follows re-derives
    everything from the current .state alone, in which these six cells - no
    longer free - simply drop out of consideration, the same way any other
    already-blocked cell does (see dissolve_blocked_paths' own docstring for
    the fuller argument, and the empirical check backing it).

    Once place_square_in_seat_closed runs, five of these six turn out to
    also be locally confirmed - each diagonally completes one of the five
    new seats it fills. (5, 2) is the exception: no new seat borders it, so
    nothing about the resulting board locally reveals it was ever doomed -
    get_blocked_links's path_id-membership check is the only thing that
    ever catches it (see do_closure's own docstring for why that's exactly
    the point of running this check at all).
    """
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    size = m.shape[0]
    border = [(i, j) for i in range(size) for j in range(size)
              if i in (0, size - 1) or j in (0, size - 1)]
    place_blocked_squares(m, border)
    find_alerts_set_links(m)
    assign_paths(m)

    dissolve_blocked_paths(m, get_blocked_links(m))

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'get_blocked_links/dissolve_blocked_paths: six cells now blocked',
                          show_links=True, show_real=True, colormap=colormap)

    # dissolve_blocked_paths already wrote the real StateEnum.blocked on each
    # of these six cells above, no separate finalization step needed - so
    # this just finds every seat (three corners blocked, one free) that a
    # newly-permanent blocked cell may have completed, and places a square
    # there, looped to a fixed point.
    place_square_in_seat_closed(m)

    # (5, 2) is the one self-contradicting cell with no local corroboration
    # at all: every other one of the six sits diagonal to a seat that just
    # got filled, so (5, 2) is the only one that stays invisible to a purely
    # local read of the resulting board - get_blocked_links's path_id check
    # was the only thing that ever caught it.

    # 4. clear every round's worth of stale bookkeeping and run the real
    # pipeline again from scratch on the resulting board.
    clear_all_but_state(m)
    do_closure(m, 'after blocking self-contradicting cells and filling seats', show=True)
