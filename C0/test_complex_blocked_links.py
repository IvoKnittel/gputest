import copy

import matplotlib.pyplot as plt

from map_of_squares import StateEnum
from representation import build_map_of_squares, map_of_squares_from_array, display_closure_step
from closure import find_alerts, link_patches, remove_blocked_links, fill_isolated_free_cells
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
    fill_isolated_free_cells(m)
    m_before = copy.deepcopy(m)
    resolve_cycles_and_centrality(m_before)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    display_closure_step(m_before, 'before remove_blocked_links', show_links=True, show_pivots=True, ax=ax_before)

    remove_blocked_links(m)

    assert m[3, 5].forces == []
    assert m[4, 4].forces == [(5, 6)]

    resolve_cycles_and_centrality(m)
    fill_isolated_free_cells(m)
    display_closure_step(m, 'after remove_blocked_links', show_links=True, show_pivots=True, ax=ax_after)
    plt.tight_layout()
    plt.show()


def test_self_loop_excluded_at_the_source():
    """A centre's own diagonal ring neighbour can, geometrically, also be one of
    its own corners - e.g. here P=(5,5) has both (4,5) and (5,4) (its ring
    positions 1 and 7, the two direct neighbours flanking ring position 0)
    blocked, completing triple (7,0,1) with (4,4) - ring position 0 - as the
    free corner. But (4,4) is *also* one of P's own four diagonal ring
    neighbours: the exact same cell plays both roles for the same centre at
    once. Without excluding that case, set_alert_chosen would link (4,4) to
    itself - a self-loop that says nothing (choosing (4,4) does not, by itself,
    oblige choosing itself again).

    set_alert_chosen's `if c_idx == d_idx: continue` guard excludes only that
    one entry, not the whole cell: (4,4) still ends up alert_chosen (it is
    genuinely the seat's occupant), and every one of P's *other* three free
    diagonal neighbours - (4,6), (6,6), (6,4) - still gets linked to (4,4) as
    normal, since none of them collide with their own corner. Only (4,4)'s
    forces, specifically, comes out empty here - not because it has nothing
    forcing it (three real forced_by entries prove otherwise), but because the
    one thing it would otherwise force is itself.
    """
    m = build_map_of_squares(9, 9)
    P = (5, 5)
    m[4, 5].state = StateEnum.blocked  # P's ring position 1 (direct)
    m[5, 4].state = StateEnum.blocked  # P's ring position 7 (direct)

    find_alerts(m)

    assert m[P].alert_blocked
    assert m[4, 4].alert_chosen and m[4, 4].forces == []
    assert set(m[4, 4].forced_by) == {(4, 6), (6, 6), (6, 4)}
    assert m[4, 6].forces == [(4, 4)]
    assert m[6, 6].forces == [(4, 4)]
    assert m[6, 4].forces == [(4, 4)]

    link_patches(m)
    assert m[4, 4].forces == []  # unchanged - link_patches no longer does anything

    display_closure_step(m, 'self-loop excluded: (4,4) is linked to by its diagonal '
                             'neighbours, never by itself', show_links=True)
