import copy

import matplotlib.pyplot as plt

from representation import map_of_squares_from_array, display_closure_step
from closure import find_alerts, link_patches, remove_blocked_links, promote_isolated_free_cells
from test_representation import resolve_cycles_and_centrality


def test_remove_blocked_links_disagreeing_removals():
    """remove_blocked_links must not require every forced_by entry to agree
    before removing a doomed target - a single genuine (live) blocker is enough,
    even when other forcers don't block the same target. This grid
    (the same one as test_link_patches_relays_past_self_loop_candidate) has two
    such disagreeing cases once find_alerts and link_patches have run:

    (3, 5).forces == [(3, 4)], forced_by == [(2, 5), (2, 7), (4, 7)]. Only
    (2, 5) blocks (3, 4) - it's one of (2, 5)'s own diagonal neighbours; (2, 7)
    and (4, 7) don't touch it at all. (2, 5) alone is enough: it's a live,
    direct forcer of (3, 5) (same patch, a real .forces edge), so if that patch
    is ever placed, (2, 5) is chosen and (3, 4) is blocked regardless of what
    (2, 7)/(4, 7) do. (3, 5).forces must end up empty, not merely unchanged
    because two out of three forcers raised no objection.

    (4, 4).forces == [(3, 6), (5, 6)], forced_by == [(2, 3), (2, 5)]. (2, 5)
    blocks only (3, 6); neither (2, 3) nor (2, 5) blocks (5, 6). Each target is
    judged independently - (3, 6) is removed, but (5, 6), which has no blocker
    at all among either forcer, must survive.
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

    assert m[3, 5].forces == [(3, 4)]
    assert set(m[3, 5].forced_by) == {(2, 5), (2, 7), (4, 7)}
    assert m[4, 4].forces == [(3, 6), (5, 6)]
    assert set(m[4, 4].forced_by) == {(2, 3), (2, 5)}

    # A deep copy, resolved independently, so the "before" panel's pivots don't
    # leave stale centrality/patch_id sitting around for the "after" panel to
    # (incorrectly) inherit once remove_blocked_links has changed the forces.
    m_before = copy.deepcopy(m)
    resolve_cycles_and_centrality(m_before)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    display_closure_step(m_before, 'before remove_blocked_links', show_links=True, show_pivots=True, ax=ax_before)

    remove_blocked_links(m)

    assert m[3, 5].forces == []
    assert m[4, 4].forces == [(5, 6)]

    resolve_cycles_and_centrality(m)
    display_closure_step(m, 'after remove_blocked_links', show_links=True, show_pivots=True, ax=ax_after)
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

    assert m[2, 5].forces == [(1, 6)]
    assert m[2, 7].forces == [(1, 6), (1, 8)]

    link_patches(m)

    assert m[1, 6].forces == [(1, 8)]

    display_closure_step(m, 'link_patches: (1,6) relays past the (2,5) self-loop to (1,8)',
                          show_links=True)
