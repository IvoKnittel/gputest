"""Written by Rose (designer role - see .claude/agents/rose.md), for Andi and Ivo.

Terminology note: "seat" = a 2x2 block with 3 corners blocked and 1 free (the free
one is where an alert_chosen item must sit) - our team's name for what closure.py's
own docstrings call "an alert" as a noun. The code identifiers themselves
(alert_blocked/alert_chosen/find_alerts) are unchanged; this is just how we talk
and write about it.

Purpose: make the seat -> forces/forced_by cascade, and the "why holes can never
happen" guarantee, visible - not to add new coverage of code paths nobody's hit yet.
Every scenario here only varies its chosen/blocked *input* (built with
representation.py's existing builders) and runs it through the existing
find_alerts/link_patches/resolve_cycles_and_centrality/check_tiling_invariant
pipeline - no production code changed, no quality/placement involved anywhere,
matching how the team is scoping this problem right now (map of chosen items in,
patches and conflicts out).

Each test also renders the scenario to docs/rose_cascades_and_holes/ via
representation.py's existing display functions, captured with plt.savefig instead
of plt.show() (this repo has no test currently exercising InvalidTilingError, or any
rendering output saved to disk - both new here, for the explainer doc these feed).

Run directly with `python3 test_rose_cascades_and_holes.py` (no pytest currently
installed in the sandbox this was written in) - every test_* function is a plain
function with plain asserts, no pytest-specific API used, so it also runs fine under
a real pytest once one's available.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from map_of_squares import StateEnum, InvalidTilingError
from representation import (build_map_of_squares,
                             build_cycle,
                             describe_links,
                             real_space_map,
                             display_closure_step,
                             display_links)
from closure import (find_alerts, link_patches, resolve_cycles_and_centrality,
                      check_tiling_invariant, place_squares)

IMG_DIR = os.path.join(os.path.dirname(__file__), "docs", "rose_cascades_and_holes")
os.makedirs(IMG_DIR, exist_ok=True)


def save(name):
    """Save whatever figure display_closure_step/display_links just drew (they call
    plt.show(), a no-op under the Agg backend, so the figure is still live) and close
    it, so figures don't pile up across tests.
    """
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def test_one_alert_is_born():
    """Minimal case: two blocked cells are all it takes for a seat to start forming.

    m[2,2] and m[3,2] blocked (e.g. as a side effect of two squares chosen
    elsewhere) is enough to threaten the quads on *both* sides of that pair at
    once: quad {(2,1),(2,2),(3,1),(3,2)} and quad {(2,2),(2,3),(3,2),(3,3)} each
    now have 2 of their 4 corners blocked - one short of becoming a real seat
    (3 blocked, 1 free). find_alerts catches both, symmetrically:
    on the left, (2,1) becomes alert_blocked, with (3,1) the seat that would
    need filling; on the right, (2,3) becomes alert_blocked, with (3,3) the
    seat. Each pair also points back at itself - (2,1) and (3,1) are a genuine
    mutual pair, each alert_blocked in its own right with the other as its own
    corner (same for (2,3)/(3,3)) - but the *link* this produces is never
    (2,1)->(3,1) itself: (2,1) is the item at risk of being *blocked*, not a
    candidate to be *chosen*, so set_alert_chosen links (2,1)'s own free
    diagonal neighbours - (1,0), (1,2), (3,0) - to (3,1) directly, not (2,1)
    (see set_alert_chosen's docstring for why). (2,1) ends up with forces == set()
    itself: nothing forces *it*, since none of its own diagonal neighbours
    happen to be alert_blocked centres of their own.
    """
    m = build_map_of_squares(7, 7)
    m[2, 2].state = StateEnum.blocked
    m[3, 2].state = StateEnum.blocked

    find_alerts(m)

    assert m[2, 1].alert_blocked and m[2, 1].forces == set()
    assert m[3, 1].alert_chosen and m[3, 1].forced_by == {(1, 0), (1, 2), (3, 0)}
    assert m[1, 0].forces == {(3, 1)} and m[1, 2].forces == {(3, 1), (3, 3)} and m[3, 0].forces == {(3, 1)}

    assert m[2, 3].alert_blocked and m[2, 3].forces == set()
    assert m[3, 3].alert_chosen and m[3, 3].forced_by == {(1, 2), (1, 4), (3, 4)}
    assert m[1, 4].forces == {(3, 3)} and m[3, 4].forces == {(3, 3)}

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, "one alert is born - after find_alerts", show_links=True, colormap=colormap)
    save("1_one_alert_is_born.png")


def test_cascade_propagates_a_second_hop():
    """A cascade that used to need link_patches now happens in a single pass of
    find_alerts alone - and link_patches, run afterward, is provably a no-op.

    Two *separate* blocked pairs are set up far apart, each threatening its own
    seat: (2,2)+(3,2) (as in test_one_alert_is_born, threatening a seat at
    (3,3)/(2,3)) and (5,4)+(5,5) (threatening one at (4,4)/(4,5)). These look
    unrelated - until you notice (3,3) and (4,4) are diagonal neighbours of each
    other. set_alert_chosen already accounts for this directly: when (2,2)+(3,2)
    are processed and (2,3) comes out alert_blocked with corner (3,3), every
    free diagonal neighbour of (2,3) - including (4,4) - gets linked straight to
    (3,3); and symmetrically, when (5,4)+(5,5) make (4,4) alert_blocked with
    corner (4,5), every free diagonal neighbour of *that* - including (3,3) -
    gets linked straight to (4,5). So (3,3).forces == {(4,5)} and
    (4,4).forces == {(2,3)} are both already true the instant find_alerts
    returns - no relay stage needed, no "replace, not merge" required (each
    only ever gets written once).

    link_patches, called afterward, changes nothing at all - not just for these
    two cells, every alert_blocked/alert_chosen/.forces/.forced_by value on the
    whole board is bit-for-bit identical before and after, which is exactly
    what link_patches's own docstring ("now a guaranteed no-op") predicts.
    """
    m = build_map_of_squares(9, 9)
    m[2, 2].state = StateEnum.blocked
    m[3, 2].state = StateEnum.blocked
    m[5, 4].state = StateEnum.blocked
    m[5, 5].state = StateEnum.blocked

    find_alerts(m)
    assert m[3, 3].alert_chosen and m[3, 3].forces == {(4, 5)}
    assert m[4, 4].alert_chosen and m[4, 4].forces == {(2, 3)}
    assert (3, 3) in m[4, 5].forced_by
    assert (4, 4) in m[2, 3].forced_by

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, "after find_alerts - already fully cascaded", show_links=True, colormap=colormap)
    save("2a_before_link_patches.png")

    # Snapshot .forces/.forced_by as fresh set copies, not just references - both
    # fields get replaced (never mutated in place) wherever link_patches touches them,
    # but a copy here is what actually makes before/after a real before/after
    # comparison rather than two views of the same live object.
    before = {(i, j): (m[i, j].alert_blocked, m[i, j].alert_chosen,
                        set(m[i, j].forces), set(m[i, j].forced_by))
              for i in range(9) for j in range(9)}
    link_patches(m)
    after = {(i, j): (m[i, j].alert_blocked, m[i, j].alert_chosen,
                       set(m[i, j].forces), set(m[i, j].forced_by))
             for i in range(9) for j in range(9)}
    assert before == after  # link_patches: a guaranteed no-op now, board-wide

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, "after link_patches - unchanged, board-wide", show_links=True, colormap=colormap)
    save("2b_after_link_patches.png")


def test_a_ring_has_no_safe_end_until_one_is_cut():
    """A patch with no terminal at all needs a different resolution.

    A chain always bottoms out at a terminal (a forces=set() item) - that's the safe
    place find_central_patch_items seeds a patch_id/centrality from. Link four
    items in a closed loop instead (nobody's forces is ever empty) and there's
    nothing to seed from: this shape needs find_cycle_patches's separate
    ring-leader election first, which finds the single largest flattened index on
    the ring and cuts its forces there - turning the ring into an ordinary chain
    with a terminal, that find_central_patch_items can then resolve normally.
    """
    m = build_map_of_squares(6, 6)
    cycle_cells = [(1, 1), (1, 3), (3, 3), (3, 1)]
    build_cycle(m, cycle_cells)

    assert all(m[c].forces for c in cycle_cells)  # closed loop: nobody is a terminal yet

    display_links(m, title="a ring, before resolution")
    save("3a_ring_before.png")

    resolve_cycles_and_centrality(m)

    # (3,3) = 3*6+3 = 21 is the largest flattened index on this ring - it's the one
    # that gets cut open into the patch's terminal.
    assert m[3, 3].forces == set() and m[3, 3].centrality == 0
    assert len({m[c].patch_id for c in cycle_cells}) == 1  # still one shared patch

    display_links(m, title="the same ring, opened at its one cut point")
    save("3b_ring_after.png")


def test_what_a_hole_actually_looks_like():
    """The negative case: what check_tiling_invariant exists to catch, and what it
    costs in real space if it's ever allowed to happen.

    Force all 4 corners of one seat blocked directly - including the seat itself,
    which should never end up anything but occupied (bypassing find_alerts
    entirely - this is deliberately the state the machinery exists to prevent ever
    being reached; find_alerts run continuously would have promised the would-be
    seat's occupant safe well before this point). check_tiling_invariant
    catches it, as it should. But the more concrete cost: real_space_map stamps a
    chosen anchor's real-space footprint from its own position and its 3
    neighbours - real-space cell (3, 3) (the shared corner of this exact quad) can
    only ever be stamped by one of these four specific anchors being chosen. With
    all four permanently blocked, that one real-space pixel is now unreachable by
    *anything*, forever - regardless of what gets chosen anywhere else on the
    grid. That single stuck pixel is the actual, physical meaning of "a hole".
    """
    m = build_map_of_squares(6, 6)
    for pos in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        m[pos].state = StateEnum.blocked

    try:
        check_tiling_invariant(m)
        raised = False
    except InvalidTilingError:
        raised = True
    assert raised, "check_tiling_invariant should have caught the all-blocked quad"

    # Surround it with ordinary, legal placements elsewhere, to show the hole
    # sitting inside an otherwise normally-tiled area, not just alone on a blank grid.
    place_squares(m, [(0, 0), (0, 4), (4, 0), (4, 4)])

    rs = real_space_map(m)
    assert rs[3, 3] == 0  # the trapped pixel: provably unreachable, no matter what

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, "a hole: real (3,3) can never be covered", show_real=True, colormap=colormap)
    save("4_the_hole.png")


if __name__ == "__main__":
    test_one_alert_is_born()
    print("test_one_alert_is_born passed")
    test_cascade_propagates_a_second_hop()
    print("test_cascade_propagates_a_second_hop passed")
    test_a_ring_has_no_safe_end_until_one_is_cut()
    print("test_a_ring_has_no_safe_end_until_one_is_cut passed")
    test_what_a_hole_actually_looks_like()
    print("test_what_a_hole_actually_looks_like passed")
    print("\nall 4 scenarios passed, images written to docs/rose_cascades_and_holes/")
