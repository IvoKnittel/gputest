import matplotlib.pyplot as plt

from map_of_squares import StateEnum
from representation import build_map_of_squares, set_link, display_closure_step
from closure import find_alerts, link_patches, remove_blocked_links, resolve_cycles_and_centrality


def test_remove_blocked_links_deletes_self_blocking_same_patch():
    """A patch that turns out to be self-blocking - one of its own members
    diagonally blocks another member of the same patch, so the patch can never
    actually be chosen - gets unwound by remove_blocked_links (see
    closure.mark_self_blocking_same_patch / closure.propagate_deletions_once),
    not merely have one offending link quietly pruned the way the old,
    forced_by-diagonal-based remove_blocked_links used to.

    Hand-built with the link library (set_link), the same shape as the
    (8,7)/(6,6)/(7,5) situation found in test_impossible.py::test_try_3x3_hole3
    round 9: (8, 7) forces both (6, 6) and (7, 5) - but (6, 6) and (7, 5) are
    diagonal neighbours of each other (offset (1, -1)), so choosing either one
    blocks the other outright. Once patch_id converges, (8, 7)/(6, 6)/(7, 5) all
    end up sharing one patch_id (closure.propagate_patch_id_from_roots, since
    (8, 7) is the root here - nothing forces it) - exactly the self-blocking
    shape remove_blocked_links exists to catch.
    """
    m = build_map_of_squares(12, 12)
    set_link(m, (8, 7), (6, 6))
    set_link(m, (8, 7), (7, 5))

    resolve_cycles_and_centrality(m)

    patch_id = m[8, 7].patch_id
    assert patch_id != -1
    assert m[6, 6].patch_id == patch_id
    assert m[7, 5].patch_id == patch_id

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    display_closure_step(m, 'before remove_blocked_links', show_links=True, show_pivots=True, show_real=True, ax=ax_before)

    remove_blocked_links(m)

    # (6, 6) and (7, 5) - the self-blocking pair - are fully unwound: back to
    # plain free items, no longer forcing or forced by anything.
    for pos in [(6, 6), (7, 5)]:
        item = m[pos]
        assert item.state == StateEnum.free
        assert item.patch_id == -1
        assert item.forces == set()
        assert item.forced_by == set()
        assert not item.delete

    # (8, 7) survives - it never itself diagonally blocked anything - but both
    # of its links to the deleted pair are gone; its own .patch_id is left
    # stale (remove_blocked_links only resets a deleted item's own patch_id,
    # never a surviving forcer's - see propagate_deletions_once's docstring).
    assert m[8, 7].alert_chosen
    assert m[8, 7].forces == set()
    assert m[8, 7].patch_id == patch_id

    display_closure_step(m, 'after remove_blocked_links', show_links=True, show_pivots=True, show_real=True, ax=ax_after)
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
    assert m[4, 4].alert_chosen and m[4, 4].forces == set()
    assert m[4, 4].forced_by == {(4, 6), (6, 6), (6, 4)}
    assert m[4, 6].forces == {(4, 4)}
    assert m[6, 6].forces == {(4, 4)}
    assert m[6, 4].forces == {(4, 4)}

    link_patches(m)
    assert m[4, 4].forces == set()  # unchanged - link_patches no longer does anything

    display_closure_step(m, 'self-loop excluded: (4,4) is linked to by its diagonal '
                             'neighbours, never by itself', show_links=True)
