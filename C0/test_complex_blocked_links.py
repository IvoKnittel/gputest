import numpy as np
import matplotlib.pyplot as plt

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, place_blocked_squares, display_closure_step
from closure import reset_alert_bookkeeping, find_alerts, assign_paths, get_blocked_links, set_blocked_links, finalize_blocked_tmp, place_square_in_seat_closed, do_closure, resolve_cycles_and_centrality

def test_two_forced_cells_block_each_other():
    """A 12x12 map: a 1-cell blocked margin wrapped around the 10x10 area that used
    to be the whole map, so the forces/path_id behaviour under test comes from the
    shape itself, not from free cells touching the array's own edge.
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
    set_blocked_links(m, get_blocked_links(m))
    resolve_cycles_and_centrality(m)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'after', show_links=True, show_real=True, ax=ax_before, colormap=colormap)
    reset_alert_bookkeeping(m)

    find_alerts(m)
    set_blocked_links(m, get_blocked_links(m))
    resolve_cycles_and_centrality(m)
    
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'closure redone', show_links=True, show_real=True, ax=ax_before, colormap=colormap)


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
    indistinguishable from a genuinely blocked cell - and strips every one
    of the six ids back out of every cell's path_id, marked or not, since a
    broken path shouldn't still claim members anywhere on the board.
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

    # 4. clear every round's worth of stale bookkeeping and run the real
    # pipeline again from scratch on the resulting board.
    reset_alert_bookkeeping(m)
    do_closure(m, 'after finalizing blocked_tmp and filling seats', show=True)
