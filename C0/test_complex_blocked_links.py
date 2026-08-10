import numpy as np
import matplotlib.pyplot as plt

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, display_closure_step
from closure import redo_closure, reset_alert_bookkeeping, find_alerts, link_patches, remove_blocked_links, resolve_cycles_and_centrality

def test_two_forced_cells_block_each_other():

    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    find_alerts(m)
    link_patches(m)
    resolve_cycles_and_centrality(m)

    # (4, 3) and (6, 5) both force the mutually-blocking pair (5, 3)/(4, 4) -
    # common forcers of a self-blocking pair get clogged by remove_blocked_links,
    # but clogging must not touch .patch_id (see closure.clog_item).
    assert {(5, 3), (4, 4)} <= m[4, 3].forces
    assert {(5, 3), (4, 4)} <= m[6, 5].forces
    patch_id_4_3 = m[4, 3].patch_id
    patch_id_6_5 = m[6, 5].patch_id

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'before remove_blocked_links', show_links=True, show_pivots=True, show_real=True, ax=ax_before, colormap=colormap)

    remove_blocked_links(m)

    assert m[4, 3].patch_id == patch_id_4_3
    assert m[6, 5].patch_id == patch_id_6_5

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'after remove_blocked_links', show_links=True, show_pivots=True, show_real=True, ax=ax_before, colormap=colormap)
    reset_alert_bookkeeping(m, keep_patch_id_for_blocked=True)

    find_alerts(m); link_patches(m)
    resolve_cycles_and_centrality(m)
    
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'closure redone', show_links=True, show_pivots=True, show_real=True, ax=ax_before, colormap=colormap)
