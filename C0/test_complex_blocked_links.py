import numpy as np
import matplotlib.pyplot as plt

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, place_blocked_squares, display_closure_step
from closure import do_closure, reset_alert_bookkeeping, find_alerts, remove_blocked_links, resolve_cycles_and_centrality

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
    remove_blocked_links(m)
    resolve_cycles_and_centrality(m)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'after', show_links=True, show_real=True, ax=ax_before, colormap=colormap)
    reset_alert_bookkeeping(m)

    find_alerts(m)
    remove_blocked_links(m)
    resolve_cycles_and_centrality(m)
    
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'closure redone', show_links=True, show_real=True, ax=ax_before, colormap=colormap)


def test_remove_blocked_links_marks_blocked_tmp():
    """Same shape as test_two_forced_cells_block_each_other, but stopping
    right after remove_blocked_links (before resolve_cycles_and_centrality/
    assign_paths get anywhere near path_id) to look at what the new
    forced_by/.forces-only conflict detection actually marks. Every
    StateEnum.blocked_tmp cell shows up red.

    (5, 4) and (6, 4) diagonally block each other and are both independently
    forced - the seed pair phase 1 detects - but only (5, 4) itself ends up
    blocked_tmp: contradiction is decided by comparing pairs of a cell's own
    .forces targets, not by the size of the cell's own accumulated path_id.
    (5, 4) forces both (5, 5) and (6, 4), and those two end up sharing an id
    (both {90, 76}) - a real disagreement between two things (5, 4) would
    transitively commit to - so (5, 4) is contradictory. (6, 4) forces (4, 5)
    (empty path_id) and (6, 5) ({90}) - no intersection between its own two
    targets, and neither target is itself blocked_tmp - so (6, 4) doesn't
    qualify, even though it separately carries a multi-id path_id of its own.
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
    remove_blocked_links(m)

    blocked_tmp = [(i, j) for i in range(size) for j in range(size)
                   if m[i, j].state == StateEnum.blocked_tmp]
    assert (5, 4) in blocked_tmp
    assert (6, 4) not in blocked_tmp

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'remove_blocked_links: blocked_tmp in red',
                          show_links=True, show_real=True, colormap=colormap)
