import numpy as np

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, place_blocked_squares, display_closure_step
from closure import reset_alert_bookkeeping, find_alerts, assign_paths, get_blocked_links, set_blocked_links, finalize_blocked_tmp, place_square_in_seat_closed, check_tiling_invariant, do_closure

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
    - that test owns the specific get_blocked_links/set_blocked_links
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
    find_alerts(m)
    # Only the two hand-placed dominoes are chosen yet, and .forces/.forced_by
    # (find_alerts's own output) are already real - path_id/centrality
    # (assign_paths) and the blocked-links procedure (get_blocked_links/
    # set_blocked_links) haven't run yet, so this is the alert/link structure
    # in its rawest, pre-path form.
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'initial alerts/links, before blocked-links',
                          show_links=True, show_real=True, colormap=colormap)

    assign_paths(m)

    set_blocked_links(m, get_blocked_links(m))
    finalize_blocked_tmp(m)

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
    reset_alert_bookkeeping(m)

    find_alerts(m)
    assign_paths(m)
    p2 = get_blocked_links(m)
    # The board is already a fixed point of one round - redoing it from a
    # clean bookkeeping slate finds no further contradictions.
    assert p2 == set()
    set_blocked_links(m, p2)
    finalize_blocked_tmp(m)
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
    right after get_blocked_links/set_blocked_links to look at what they
    actually mark. Both need path_id to already be real (run after
    assign_paths, do_closure's own order) - calling get_blocked_links straight
    after find_alerts, without assign_paths in between, leaves every path_id
    empty and returns an empty set too, not because nothing is wrong but
    because there was nothing yet for it to check. Every StateEnum.blocked_tmp
    cell shows up red.

    get_blocked_links(m) returns {26, 33, 62, 64, 90, 117} - six
    self-contradicting ids, found across the whole board, not just the two
    hand-placed squares' immediate neighbourhood. Each is a cell's own
    unique_id, shared with one of its own free diagonal neighbours: e.g. 64
    is (5, 4)'s own id - one of its diagonal neighbours ((4, 5), (6, 5), or
    (6, 3), all three) carries 64 too, so choosing (5, 4) would block a
    fellow member of its own path. set_blocked_links then marks each id's
    origin cell (the one whose own unique_id it is) blocked_tmp:
    (2, 2), (2, 9), (5, 2), (5, 4), (7, 6), (9, 9) - clears every one of
    those six cells' own .forces/.forced_by so they end up structurally
    indistinguishable from a genuinely blocked cell, retracts them from
    every other cell's .forces/.forced_by too (no dangling links anywhere
    on the board, not just locally), and strips every one of the six ids
    back out of every cell's path_id, marked or not, since a broken path
    shouldn't still claim members anywhere on the board.

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
    find_alerts(m)
    assign_paths(m)

    p = get_blocked_links(m)
    assert p == {26, 33, 62, 64, 90, 117}

    set_blocked_links(m, p)

    blocked_tmp = {(i, j) for i in range(size) for j in range(size)
                   if m[i, j].state == StateEnum.blocked_tmp}
    assert blocked_tmp == {(2, 2), (2, 9), (5, 2), (5, 4), (7, 6), (9, 9)}

    for pos in blocked_tmp:
        assert m[pos].forces == set()
        assert m[pos].forced_by == set()

    for i in range(size):
        for j in range(size):
            assert not (m[i, j].path_id & p)
            # No other cell is left pointing at or from a role one of these
            # six no longer plays either - set_blocked_links retracts both
            # sides of every severed link, not just the blocked_tmp cell's
            # own two sets.
            assert not (blocked_tmp & m[i, j].forces)
            assert not (blocked_tmp & m[i, j].forced_by)

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'get_blocked_links/set_blocked_links: blocked_tmp in red',
                          show_links=True, show_real=True, colormap=colormap)

    # 1. blocked_tmp is provisional - make it permanent now that nothing
    # further will change based on the flag.
    finalize_blocked_tmp(m)
    assert not any(m[i, j].state == StateEnum.blocked_tmp
                   for i in range(size) for j in range(size))

    # 2-3. find every seat (three corners blocked, one free) that a newly-
    # permanent blocked cell may have completed, and place a square there -
    # place_square_in_seat_closed does both, looped to a fixed point.
    place_square_in_seat_closed(m)

    # (5, 2) is the one self-contradicting cell with no local corroboration
    # at all: every other one of the six sits diagonal to a seat that just
    # got filled, so (5, 2) is the only one that stays invisible to a purely
    # local read of the resulting board - get_blocked_links's path_id check
    # was the only thing that ever caught it.
    new_chosen = {(i, j) for i in range(size) for j in range(size)
                  if m[i, j].state == StateEnum.chosen} - {(4, 7), (5, 7), (8, 3), (8, 4)}
    DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    locally_confirmed = {pos for pos in blocked_tmp
                          if any((pos[0] + di, pos[1] + dj) in new_chosen
                                 for di, dj in DIAGONAL)}
    assert blocked_tmp - locally_confirmed == {(5, 2)}

    # 4. clear every round's worth of stale bookkeeping and run the real
    # pipeline again from scratch on the resulting board.
    reset_alert_bookkeeping(m)
    do_closure(m, 'after finalizing blocked_tmp and filling seats', show=True)
