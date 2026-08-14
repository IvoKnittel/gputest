import numpy as np
import matplotlib.pyplot as plt

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, place_blocked_squares, display_closure_step
from closure import redo_closure, reset_alert_bookkeeping, find_alerts, remove_blocked_links, resolve_cycles_and_centrality

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
    resolve_cycles_and_centrality(m)

    # (5, 4) and (7, 6) both force the mutually-blocking pair (6, 4)/(5, 5) -
    # common forcers of a self-blocking pair get clogged by remove_blocked_links,
    # but clogging must not touch .path_id (see closure.clog_item).
    assert {(6, 4), (5, 5)} <= m[5, 4].forces
    assert {(6, 4), (5, 5)} <= m[7, 6].forces
    path_id_5_4 = m[5, 4].path_id
    path_id_7_6 = m[7, 6].path_id

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'before remove_blocked_links', show_links=True, show_pivots=True, show_real=True, ax=ax_before, colormap=colormap)

    remove_blocked_links(m)

    assert m[5, 4].path_id == path_id_5_4
    assert m[7, 6].path_id == path_id_7_6

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'after remove_blocked_links', show_links=True, show_pivots=True, show_real=True, ax=ax_before, colormap=colormap)
    reset_alert_bookkeeping(m, keep_path_id_for_blocked=True)

    find_alerts(m)
    resolve_cycles_and_centrality(m)
    
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'closure redone', show_links=True, show_pivots=True, show_real=True, ax=ax_before, colormap=colormap)
