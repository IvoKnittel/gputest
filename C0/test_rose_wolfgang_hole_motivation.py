"""Written by Rose and Wolfgang, for Andi and Ivo.

Ivo pointed at test_impossible.py::test_other_full_2x2 - a placement that "looks
harmless" (four chosen squares, no two of them diagonal neighbours of each other,
so place_squares raises no complaint about any of them individually) but leaves
(5,5) unreachable. Task: motivate the no-fully-blocked-2x2 rule by naively placing
squares and showing (5,5) is unreachable - not hand-forcing .state like
test_rose_cascades_and_holes.py's hole demo does.

Originally two scenarios: a naive one-shot batch placement (closure never gets a
turn to run at all), and placing the same four squares one at a time with
find_alerts/link_patches/remove_blocked_links run after each - which correctly
flagged (5,5) as alert_chosen well before the danger was complete, but did nothing
to stop the fourth placement from blocking it anyway. That second finding is what
motivated closure.forced_closure (see Andi/Ivo): a placement's own recorded .forces
were never being acted on, only ever left as a passive record for later. The one
test left in this file now applies that rule and shows what it actually changes.
"""

import os

import matplotlib.pyplot as plt

from map_of_squares import StateEnum, InvalidTilingError
from representation import build_map_of_squares, place_squares, display_closure_step
from closure import find_alerts, link_patches, remove_blocked_links, forced_closure, reset_alert_bookkeeping

IMG_DIR = os.path.join(os.path.dirname(__file__), "docs", "rose_cascades_and_holes")
os.makedirs(IMG_DIR, exist_ok=True)


def save(name):
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def test_continuous_closure_with_forced_placement_catches_it_early():
    """The same four squares, placed one at a time - but now acting on .forces
    immediately after each placement (closure.forced_closure), instead of only
    ever recording it: after find_alerts/link_patches/remove_blocked_links, if
    the square just placed already had its own .forces on it (from an *earlier*
    round's closure pass), every position that chain transitively reaches gets
    placed too, right then, before moving on to the next manual placement. Since
    that can place more than one square in a single round, all of a round's
    alert/link bookkeeping (closure.reset_alert_bookkeeping) is then thrown away
    and find_alerts/link_patches/remove_blocked_links re-run from scratch against
    the new board state, instead of leaving the newly-placed cells stuck showing
    a stale alert_chosen=True from before they were placed - and only then is the
    round displayed.

    Turns out two of the four "manual" placements below were never actually free
    choices by the time they're made: (5,6) already has .forces == [(3,5), (5,5)]
    the instant it's placed (round 2's closure pass flagged it - unnoticed by the
    original version of this test, which just placed it anyway). Chasing that
    chain immediately drags (5,5) and (3,5) themselves into StateEnum.chosen -
    not just alert_chosen - fulfilling (5,5)'s own promise on the spot rather
    than leaving it sitting there as a risk. By the time (6,6) is placed next,
    (5,5) is no longer free to break - it's a diagonal neighbour of an
    already-*chosen* square, so place_squares's own diagonal-overlap check
    (real_space_map) rejects it outright with InvalidTilingError, before any
    state gets corrupted.

    This is not a general proof that acting on .forces always prevents a
    promise from being broken - which square gets placed in which order is
    still an external choice this test makes by hand, and a different order
    could still let an unrelated placement block a promise before its own
    forces chain ever gets a turn to fire (see Wolfgang's notes: recording a
    promise correctly is necessary but not sufficient by itself). What this
    shows is narrower but real: for *this* scenario, the step the original
    version of this test was missing - actually acting on what closure already
    knew - would have caught the problem before it happened, not after.
    """
    m = build_map_of_squares(9, 9)

    def place_and_chase(pos, title):
        place_squares(m, [pos])
        find_alerts(m); link_patches(m); remove_blocked_links(m)
        forced = forced_closure(m, pos)
        if forced:
            # More than one cell changes state in this round (pos plus every
            # position forced_closure found) - the usual incremental "one
            # placement, one find_alerts/link_patches pass" discipline no longer
            # applies, so reset every cell's alert bookkeeping and recompute it
            # from scratch against the new board state, rather than layering new
            # flags on top of stale ones. Without this, (5,6)/(5,5)/(3,5) would
            # still carry .alert_chosen=True from before they were placed, and
            # display as "promised, still free" (yellow) instead of actually
            # chosen (cyan).
            place_squares(m, list(forced))
            reset_alert_bookkeeping(m)
            find_alerts(m); link_patches(m); remove_blocked_links(m)
        display_closure_step(m, title, show_links=True, show_real=True)

    place_and_chase((3, 3), "round 1: (3,3) placed")
    place_and_chase((4, 3), "round 2: (4,3) placed - (5,5) now promised (alert_chosen)")

    # Two corners of the quad are blocked now - (4,4) and (5,4) - and closure has
    # already, correctly, flagged (5,5) as alert_chosen: a recorded promise that it
    # must be the one to end up chosen, seeded before the danger is even complete.
    assert m[4, 4].state == StateEnum.blocked
    assert m[5, 4].state == StateEnum.blocked
    assert m[5, 5].alert_chosen and m[5, 5].state == StateEnum.free

    # (5,6) already has .forces == [(3,5), (5,5)] from round 2's closure pass,
    # before it's even placed - place_and_chase places (5,5) (and (3,5)) too,
    # in the same round, as soon as (5,6) lands.
    assert m[5, 6].forces == [(3, 5), (5, 5)]
    place_and_chase((5, 6), "round 3: (5,6) placed - its own .forces chased, (5,5) chosen on the spot")
    assert m[5, 5].state == StateEnum.chosen and m[3, 5].state == StateEnum.chosen

    # Chosen, not just alert_chosen - reset_alert_bookkeeping cleared the stale
    # flags these three carried from before they were placed, so they don't
    # still render as "promised, still free" (yellow) now that they're cyan.
    assert not m[5, 6].alert_chosen and not m[5, 5].alert_chosen and not m[3, 5].alert_chosen

    # (6,6) is now a diagonal neighbour of an already-chosen square - place_squares
    # rejects it immediately, instead of silently letting the promise break.
    try:
        place_and_chase((6, 6), "round 4: (6,6) placed anyway")
        raised = False
    except InvalidTilingError as e:
        raised = True
        # show_real=False here: (6,6) is left StateEnum.chosen in m (place_squares
        # mutates before it checks - see its docstring), so real_space_map would
        # just raise the same error again trying to render it.
        display_closure_step(m, f"round 4: place_squares rejected (6,6) - {e}",
                              show_links=True, show_real=False)
    assert raised, "(6,6) should be rejected outright now that (5,5) is already chosen"


if __name__ == "__main__":
    test_continuous_closure_with_forced_placement_catches_it_early()
    print("test_continuous_closure_with_forced_placement_catches_it_early passed")
