import copy

import numpy as np
import matplotlib.pyplot as plt

from representation import (map_of_squares_from_array,
                             display_map_of_squares_3States,
                             real_space_map,
                             colorize,
                             colorize_with_alerts,
                             FREE_COLOR,
                             CHOSEN_COLOR,
                             BLOCKED_COLOR,
                             MARGIN_COLOR)
from closure import (promote_isolated_free_cells,
                      find_alerts,
                      link_patches,
                      remove_blocked_links,
                      copy_map_reverse,
                      check_tiling_invariant,
                      find_cycle_patches,
                      find_central_patch_items)


def grid_on(ax, rows, cols, offset=-0.5):
    """Draw a line between every cell of a rows x cols image shown via imshow on ax
    - like MATLAB's grid on, but at cell boundaries rather than through the tick
    labels, so each field is visibly boxed off from its neighbours.

    offset is where the first boundary line sits relative to the image's own index
    origin: -0.5 (the default) matches a plain, unshifted imshow; pass 0 for an
    imshow whose extent has been shifted by +0.5, as real_space_map's grid
    intersections require (see test_map_input_display).

    Drawn as explicit lines (vlines/hlines) rather than via minor-tick gridlines,
    since minor ticks that land exactly on major tick positions (as offset=0 does)
    are not reliably drawn as gridlines across matplotlib versions/backends.
    """
    x_lines = np.arange(offset, cols + offset + 1, 1)
    y_lines = np.arange(offset, rows + offset + 1, 1)
    ax.vlines(x_lines, y_lines[0], y_lines[-1], color='black', linewidth=1)
    ax.hlines(y_lines, x_lines[0], x_lines[-1], color='black', linewidth=1)


def test_map_input_display():
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)

    fig, (ax_real, ax_map) = plt.subplots(1, 2, figsize=(10, 5))

    real_display = real_space_map(m)
    ax_real.imshow(colorize(real_display, {0: FREE_COLOR, 1: CHOSEN_COLOR}))
    ax_real.set_title('real space')
    ax_real.axis('on')
    grid_on(ax_real, *real_display.shape)

    # map_of_squares is isomorphic to the grid-intersection points of real_space:
    # the centre of square (i, j) is exactly the corner shared by the four
    # real-space cells (i,j), (i,j+1), (i+1,j), (i+1,j+1) that square occupies -
    # i.e. real-space coordinate (j + 0.5, i + 0.5). Shifting this image's extent
    # by +0.5 puts every cell centre there, and matching this axes' limits to
    # ax_real's then draws it inset within the very same frame - one cell-width
    # smaller and centred on all sides, since map_of_squares has one fewer row and
    # column than real_space.
    map_rows, map_cols = m.shape
    map_display = display_map_of_squares_3States(m)
    ax_map.set_facecolor(MARGIN_COLOR)
    ax_map.imshow(colorize(map_display, {0: FREE_COLOR, 1: CHOSEN_COLOR, -1: BLOCKED_COLOR}),
                  extent=(0, map_cols, map_rows, 0))
    ax_map.set_title('map_of_squares')
    ax_map.axis('on')
    grid_on(ax_map, map_rows, map_cols, offset=0)
    ax_map.set_xlim(ax_real.get_xlim())
    ax_map.set_ylim(ax_real.get_ylim())

    plt.tight_layout()
    plt.show()


def test_4links_situation():
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)

    fig, (ax_real, ax_map) = plt.subplots(1, 2, figsize=(10, 5))

    real_display = real_space_map(m)
    ax_real.imshow(colorize(real_display, {0: FREE_COLOR, 1: CHOSEN_COLOR}))
    ax_real.set_title('real space')
    ax_real.axis('on')
    grid_on(ax_real, *real_display.shape)

    # map_of_squares is isomorphic to the grid-intersection points of real_space:
    # the centre of square (i, j) is exactly the corner shared by the four
    # real-space cells (i,j), (i,j+1), (i+1,j), (i+1,j+1) that square occupies -
    # i.e. real-space coordinate (j + 0.5, i + 0.5). Shifting this image's extent
    # by +0.5 puts every cell centre there, and matching this axes' limits to
    # ax_real's then draws it inset within the very same frame - one cell-width
    # smaller and centred on all sides, since map_of_squares has one fewer row and
    # column than real_space.
    map_rows, map_cols = m.shape
    map_display = display_map_of_squares_3States(m)
    ax_map.set_facecolor(MARGIN_COLOR)
    ax_map.imshow(colorize(map_display, {0: FREE_COLOR, 1: CHOSEN_COLOR, -1: BLOCKED_COLOR}),
                  extent=(0, map_cols, map_rows, 0))
    ax_map.set_title('map_of_squares')
    ax_map.axis('on')
    grid_on(ax_map, map_rows, map_cols, offset=0)
    ax_map.set_xlim(ax_real.get_xlim())
    ax_map.set_ylim(ax_real.get_ylim())

    plt.tight_layout()
    plt.show()

def display_closure_step(m, title, show_links=False, show_pivots=False, ax=None):
    """Show a single map_of_squares panel coloured via colorize_with_alerts, so
    alert_blocked (blue), alert_chosen (yellow), and both-at-once (green) are
    visible on top of the plain free/chosen/blocked colours - see
    test_do_closure_steps, which calls this after each do_closure stage in turn.

    show_links=True additionally draws a black arrow into every alert_chosen
    item from each item in its .links (see SquareItem.links) -
    drawn from that source, not read off the target's own .reverse_links, so the arrow
    points the way the causal flow actually runs: source is what gets chosen
    (or blocked) first, and that is what forces this item, not the other way
    around. An item can have more than one, e.g. test_4links_situation), drawn
    directly on top of the image - the same (row, col) -> (x, y) = (col, row)
    mapping imshow already uses for the image itself, so no coordinate
    translation is needed. A self-loop (a source equal to (i, j) itself) has no
    direction to draw an arrow along, so it's circled in red instead -
    resolve_chosen_link (alert_graphs.py) skips self-loop candidates when it
    can, so this should only ever show up in a hand-built scenario, not one
    produced by find_alerts/link_patches - see
    test_link_patches_relays_past_self_loop_candidate in this file.

    show_pivots=True additionally draws a blue arrow, source to terminal, for
    every (source, terminal) pair compute_blue_arrows finds - source to
    terminal, not the other way around, to match the same causal direction the
    black .reverse_links/.links arrows point in: a terminal is what every item
    upstream of it, transitively, forces. Unlike drawing only from each patch's
    single furthest ("pivot") item, this draws one arrow per item that resolves
    to a terminal at all - see compute_blue_arrows for why, and
    test_remove_blocked_links_disagreeing_removals, where an intermediate item
    like (6, 4) - not just the patch's deepest item - gets its own arrow too.

    ax=None (default): create a new figure/axes and call plt.show() once this
    panel is drawn, as a standalone display. Pass an existing ax to draw into it
    instead, without creating a figure or calling plt.show() - e.g. to place two
    of these panels side by side in one figure, see
    test_do_closure_steps_reverse_check.
    """
    rows, cols = m.shape
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor(MARGIN_COLOR)
    ax.imshow(colorize_with_alerts(m))
    ax.set_title(title)
    ax.axis('on')
    grid_on(ax, rows, cols)

    if show_links:
        for i in range(rows):
            for j in range(cols):
                item = m[i, j]
                if not (item.alert_chosen and item.links):
                    continue
                for si, sj in item.links:
                    if (si, sj) == (i, j):
                        ax.plot(j, i, 'o', markersize=16, markerfacecolor='none',
                                markeredgecolor='red', markeredgewidth=2, zorder=4)
                        continue
                    # Shift both endpoints sideways, perpendicular to this
                    # arrow's own direction, so several arrows converging on
                    # the same cell from different directions (e.g. multiple
                    # items linking to the same target) fan out instead of
                    # all landing on the exact same point - see
                    # test_do_closure_steps, where (5, 2), (7, 2), (7, 3), and
                    # (7, 5) all separately link into the (6, 3)/(6, 4) pair.
                    dx, dy = j - sj, i - si
                    length = (dx ** 2 + dy ** 2) ** 0.5
                    perp_x, perp_y = (-dy / length, dx / length) if length else (0, 0)
                    offset = 0.12
                    ax.annotate('', xy=(j + perp_x * offset, i + perp_y * offset),
                                xytext=(sj + perp_x * offset, si + perp_y * offset),
                                arrowprops=dict(arrowstyle='->', color='black',
                                                 shrinkA=8, shrinkB=8,
                                                 connectionstyle='arc3,rad=0.15'),
                                zorder=3)

    if show_pivots:
        for (pi, pj), (ti, tj) in compute_blue_arrows(m):
            if (pi, pj) == (ti, tj):
                continue
            # Shift both endpoints sideways, perpendicular to the
            # source->terminal direction, so this arrow is drawn next to -
            # rather than directly on top of - a black .links arrow that
            # happens to run between (or through) the same two cells.
            dx, dy = tj - pj, ti - pi
            length = (dx ** 2 + dy ** 2) ** 0.5
            perp_x, perp_y = (-dy / length, dx / length) if length else (0, 0)
            offset = 0.25
            ax.annotate('', xy=(tj + perp_x * offset, ti + perp_y * offset),
                        xytext=(pj + perp_x * offset, pi + perp_y * offset),
                        arrowprops=dict(arrowstyle='->', color='blue',
                                         linewidth=2, shrinkA=10, shrinkB=10,
                                         connectionstyle='arc3,rad=0.0'),
                        zorder=6)

    if standalone:
        plt.tight_layout()
        plt.show()


def test_do_closure_steps():
    """Run do_closure's stages one at a time on the directed-graph review grid from
    test_map_input_display, displaying the map after each so alert_blocked/
    alert_chosen items (and their overlap) are visible as they appear.
        grid = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

            
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        """

    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    m = map_of_squares_from_array(grid)
    display_closure_step(m, '0: initial state')

 #   promote_isolated_free_cells(m)
 #   display_closure_step(m, '1: promote_isolated_free_cells')

    find_alerts(m)
    display_closure_step(m, '2: find_alerts  (alert_blocked=blue, alert_chosen=yellow, both=green)')

    link_patches(m)
    display_closure_step(m, '3: link_patches', show_links=True)

    remove_blocked_links(m)
    display_closure_step(m, '3b: remove_blocked_links', show_links=True)

 #   check_tiling_invariant(m)
 #   display_closure_step(m, '4: check_tiling_invariant  (invariant held - no state change)')

    m_reverse = copy_map_reverse(m)
    resolve_cycles_and_centrality(m)
    resolve_cycles_and_centrality(m_reverse)

    fig, (ax_forward, ax_reverse) = plt.subplots(1, 2, figsize=(10, 5))
    display_closure_step(m, '5 forward: find_cycle_patches + find_central_patch_items (pivots in blue)',
                          show_links=True, show_pivots=False, ax=ax_forward)
    display_closure_step(m_reverse, '5 reversed: same, on copy_map_reverse(m)',
                          show_links=True, show_pivots=False, ax=ax_reverse)
    plt.tight_layout()
    plt.show()


def resolve_cycles_and_centrality(m, max_gens=20):
    """Run find_cycle_patches then find_central_patch_items to convergence on m,
    in place - the "remainder" of closure after link_patches, for whichever
    stages test_do_closure_steps_reverse_check needs on both the forward and
    reversed copies.

    find_cycle_patches doesn't report whether it changed anything (unlike
    find_central_patch_items), so it's simply called a generous, fixed number of
    times (max_gens) rather than looped to a real convergence check - a
    stopgap, not a guarantee, matching this whole workaround's "lazy for now"
    scope (see copy_map_reverse).
    """
    for gen in range(max_gens):
        find_cycle_patches(m, gen)

    gen = 0
    while find_central_patch_items(m, gen):
        gen += 1


def test_do_closure_steps_reverse_check():
    """test_do_closure_steps's grid has a shape find_cycle_patches doesn't
    handle: (2, 2) and (5, 3) are each part of a genuine 2-cycle with their own
    linked partner ((3, 3) and (6, 2) respectively), straight out of find_alerts
    - but they're *also* each pointed at by several other items via .reverse_links[0]
    ((2, 4), (4, 2), (4, 4) all point at (2, 2); (5, 1), (6, 4), (7, 1), (7, 3)
    all point at (5, 3)) - a "link out of a cycle" arrangement find_cycle_patches
    was never written to expect (see the "Known gap" note on its docstring):
    ring members and tails-feeding-in are the only two shapes it accounts for,
    and a node with both roles at once lets a tail's candidate id crowd out the
    ring's own.

    copy_map_reverse's lazy workaround: build a second map with every link
    pointing the other way, run the same cycle/centrality resolution on both,
    and look at both side by side. Reversing turns (2, 2)'s incoming tails into
    outgoing reverse_links instead - a shape that isn't crowded - at the cost of turning
    whatever *was* a clean tail into the same "both roles at once" shape this is
    meant to route around. Whichever of the two maps actually converges cleanly
    for a given patch is the one to trust for it; this test just displays both,
    it doesn't automate picking between them.
    """
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    promote_isolated_free_cells(m)
    find_alerts(m)
    link_patches(m)
    remove_blocked_links(m)

    m_reverse = copy_map_reverse(m)

    check_tiling_invariant(m)
    resolve_cycles_and_centrality(m)

    check_tiling_invariant(m_reverse)
    resolve_cycles_and_centrality(m_reverse)

    fig, (ax_forward, ax_reverse) = plt.subplots(1, 2, figsize=(10, 5))
    display_closure_step(m, 'forward links', show_links=True, show_pivots=False, ax=ax_forward)
    display_closure_step(m_reverse, 'reversed links', show_links=True, show_pivots=False, ax=ax_reverse)
    plt.tight_layout()
    plt.show()


def test_link_patches_relays_past_self_loop_candidate():
    """(1, 6) in this grid ends up with two qualifying diagonal neighbours once
    find_alerts has run: (2, 7), alert_blocked and linked to a genuinely
    different item (1, 8); and (2, 5), alert_blocked but linked right back to
    (1, 6) itself (they're a mutual pair straight out of find_alerts - see the
    (3, 4)/(4, 4) pair in test_do_closure_steps for the same shape of pairing -
    that also happen to be diagonal neighbours of each other, which (3, 4)/(4, 4)
    are not).

    Choosing (1, 6) blocks (2, 7); since (2, 7) is alert_blocked, that block
    would itself create a real alert unless (1, 8) is chosen too - so (1, 6)
    being alert_chosen genuinely forces (1, 8) to be chosen alongside it, and
    resolve_chosen_link must link (1, 6) to (1, 8) to record that. The other
    diagonal neighbour, (2, 5), does not add any such obligation - it only
    reflects the mutual pairing (1, 6) and (2, 5) already have from find_alerts,
    so resolve_chosen_link skips it as a self-loop candidate rather than letting
    it crowd out the real relay to (1, 8).
    """
    grid = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    promote_isolated_free_cells(m)
    find_alerts(m)

    assert m[2, 5].reverse_links == [(1, 6)]
    assert m[2, 7].reverse_links == [(1, 6), (1, 8)]

    link_patches(m)

    assert m[1, 6].reverse_links == [(1, 8)]

    display_closure_step(m, 'link_patches: (1,6) relays past the (2,5) self-loop to (1,8)',
                          show_links=True)


def test_remove_blocked_links_disagreeing_removals():
    """remove_blocked_links must not require every links entry to agree
    before removing a doomed target - a single genuine (live) blocker is enough,
    even when other reverse-linked items don't block the same target. This grid
    (the same one as test_link_patches_relays_past_self_loop_candidate) has two
    such disagreeing cases once find_alerts and link_patches have run:

    (3, 5).reverse_links == [(3, 4)], links == [(2, 5), (2, 7), (4, 7)]. Only
    (2, 5) blocks (3, 4) - it's one of (2, 5)'s own diagonal neighbours; (2, 7)
    and (4, 7) don't touch it at all. (2, 5) alone is enough: it's a live,
    direct forcer of (3, 5) (same patch, a real .reverse_links edge), so if that patch
    is ever placed, (2, 5) is chosen and (3, 4) is blocked regardless of what
    (2, 7)/(4, 7) do. (3, 5).reverse_links must end up empty, not merely unchanged
    because two out of three reverse-linked items raised no objection.

    (4, 4).reverse_links == [(3, 6), (5, 6)], links == [(2, 3), (2, 5)]. (2, 5)
    blocks only (3, 6); neither (2, 3) nor (2, 5) blocks (5, 6). Each target is
    judged independently - (3, 6) is removed, but (5, 6), which has no blocker
    at all among either reverse-linked item, must survive.
    """
    grid = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    find_alerts(m)
    link_patches(m)

    assert m[3, 5].reverse_links == [(3, 4)]
    assert set(m[3, 5].links) == {(2, 5), (2, 7), (4, 7)}
    assert m[4, 4].reverse_links == [(3, 6), (5, 6)]
    assert set(m[4, 4].links) == {(2, 3), (2, 5)}

    # A deep copy, resolved independently, so the "before" panel's pivots don't
    # leave stale centrality/patch_id sitting around for the "after" panel to
    # (incorrectly) inherit once remove_blocked_links has changed the reverse_links.
    m_before = copy.deepcopy(m)
    resolve_cycles_and_centrality(m_before)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    display_closure_step(m_before, 'before remove_blocked_links', show_links=True, show_pivots=True, ax=ax_before)

    remove_blocked_links(m)

    assert m[3, 5].reverse_links == []
    assert m[4, 4].reverse_links == [(5, 6)]

    resolve_cycles_and_centrality(m)
    display_closure_step(m, 'after remove_blocked_links', show_links=True, show_pivots=True, ax=ax_after)
    plt.tight_layout()
    plt.show()


def compute_blue_arrows(m):
    """Simplified stand-in for the patch_id/max-centrality computation
    show_pivots used to depend on, which needed find_central_patch_items to
    have fully converged - something the fan-in crowding gap on
    find_cycle_patches (see its "Known gap" note) can leave stuck indefinitely
    - and which only ever drew one arrow per patch, from its single furthest
    ("pivot") item, even when several other items link directly to the same
    terminal (see test_remove_blocked_links_disagreeing_removals: (6, 4) links
    straight to (5, 6) just as directly as the patch's actual pivot, (2, 3),
    does, but only (2, 3) got an arrow under the old computation).

    Every alert_chosen item with no .reverse_links of its own is a terminal by
    definition (nothing further is forced) - centrality 0, assigned directly
    here, with no generational propagation needed. Every other alert_chosen
    item's own arrow is then found by walking its .reverse_links[0] chain forward, one
    hop at a time, until it reaches such a terminal - and the arrow is drawn
    from that starting item straight to the terminal, not just from whichever
    item happens to be furthest along the chain. A chain that loops back on
    itself before reaching a terminal (a pure ring - see find_cycle_patches)
    has nothing to resolve to, so no arrow is produced for it.

    Mutates m (sets .centrality = 0 on every terminal found). Returns a list of
    (source, terminal) position pairs, one per alert_chosen item that resolves
    to a terminal - the blue arrows to draw.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if item.alert_chosen and not item.reverse_links:
                item.centrality = 0

    arrows = []
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if not item.alert_chosen or not item.reverse_links:
                continue
            pos = (i, j)
            seen = {pos}
            while m[pos].reverse_links:
                pos = m[pos].reverse_links[0]
                if pos in seen:
                    pos = None
                    break
                seen.add(pos)
            if pos is not None:
                arrows.append(((i, j), pos))
    return arrows
