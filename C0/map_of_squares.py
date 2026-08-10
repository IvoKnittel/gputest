from enum import Enum
from dataclasses import dataclass, field

import numpy as np

v = 0
h = 1


class StateEnum(Enum):
    free = 0
    chosen = 1
    blocked = 2


@dataclass
class SquareItem:
    """One block-position's placement state.

    quality:    score used to rank candidate placements.
    state:      current placement state (free / chosen / blocked).
    alert_chosen:  raised by a neighbouring tile's placement pass instead of writing
                   .state directly; resolved into real state via forced_closure +
                   place_squares (see test_utils.place_and_chase).
    alert_blocked: same as alert_chosen, for the blocked outcome.
    forces:     every (row, col) index this item forces - i.e. what must also be
                chosen if this item is chosen; empty means this item forces
                nothing (a "terminal" item, in patch terms). A set, not a list -
                membership is all that ever matters here, never order, and the
                same (row, col) index is never a meaningful duplicate of itself.
                Set by alert_graphs.set_alert_chosen: for an alert_blocked
                centre P, every one of P's own free diagonal neighbours (not P
                itself - P is the item at risk of being *blocked*, never the one
                being chosen here) gets P's corner(s) added to its own .forces.
                A set rather than a single value because one item can legitimately
                force more than one other at once - it can be a free diagonal
                neighbour of several different alert_blocked centres at once (see
                test_complex_graphs.test_tree_fan_out), or a single centre can have
                more than one corner of its own - and none of them should be
                silently discarded in favour of another. A cut (find_central_patch_items,
                find_cycle_patches) clears this back to set(), severing the pairing
                entirely. Where code needs just one representative target (e.g. to
                walk a single chain for display, or to check "does this reach a
                terminal") it takes an arbitrary entry (next(iter(...))) - stable for
                the duration of one pass since nothing mutates .forces mid-pass, but
                not any particular "first" one the way a list index would be.
    forced_by:  every (row, col) index that has ever forced this item - i.e. the
                other side of the same relationship .forces records, filled
                alongside it wherever a .forces entry is created (set_alert_chosen,
                representation.set_link) so it's always available without
                needing a separate reversed copy of the map to look it up.
                Also a set, for the same reason .forces is. A .forces cut
                (find_central_patch_items, find_cycle_patches) does not retract,
                though - a stale entry can still be left behind that way.
    centrality: distance, in .forces hops, from this item's chain to the closest
                terminal (no-.forces) item of its patch; assigned by
                closure.find_central_patch_items one generation at a time, -1
                while unassigned.
    patch_id:   id of the patch this item belongs to. A "patch" is a connected
                group of alert_chosen items linked together - our actual
                building block once seats start linking squares together, as
                opposed to a lone square. Seeded at a terminal (its own
                flattened index) and propagated outward one generation at a
                time by closure.find_central_patch_items; -1 while unassigned.
                Two patches merging get reconciled to whichever id is larger;
                a patch looping back on its own id instead has its closing
                .forces cut, so cycles don't propagate forever. (This field used
                to be called path_id, back when we still called this concept
                a "path" rather than a "patch".)
    max_id:     candidate for the largest flattened index among a pure ring's
                members (one with no terminal anywhere, so centrality/patch_id
                never reach it) - used by closure.find_cycle_patches as a
                parallel-safe way to agree on a single break point without
                relying on processing order; -1 while unassigned, and reset
                back to -1 as soon as this item (or the item it forces) gets a
                real centrality, since max_id has no meaning outside a pure ring.
                A single scalar, so more than one item pointing at this one via
                .forces at once (see .forces above and test_do_closure_steps)
                can still have one candidate crowd out another here - see the
                "Known gap" note on find_cycle_patches.
    rectangle:  (row, col) index of the direct neighbour this item has been
                paired with into a domino, or (-1, -1) while unpaired. Set by
                set_square_chosen: when an item is chosen, the first direct
                neighbour found that is itself already chosen and still
                unpaired becomes its partner, both ways. Deliberately
                first-come: a chosen neighbour that already has a partner is
                skipped rather than repaired, so pairing one more chosen item
                onto either end of an existing domino later never disturbs it.
    """
    quality: float = -1.0
    state: StateEnum = StateEnum.free
    alert_chosen: bool = False
    alert_blocked: bool = False
    forces: set = field(default_factory=set)
    forced_by: set = field(default_factory=set)
    centrality: int = -1
    patch_id: int = -1
    max_id: int = -1
    rectangle: tuple = (-1, -1)


# The four direct (orthogonal) neighbours - as opposed to a diagonal one -
# checked by set_square_chosen when pairing a newly-chosen item into a domino.
DIRECT_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def set_square_chosen(map_of_squares, pos):
    """Wrapper around setting map_of_squares[pos].state to StateEnum.chosen,
    that also pairs it into a rectangle (domino) with a direct neighbour, if
    one qualifies. Lives here, not in closure.py/representation.py, so both
    (representation.py can't import from closure.py without a cycle) can call
    it as the one place .state ever becomes StateEnum.chosen.

    Checks all four direct neighbours of pos: the first one found that is
    itself already chosen *and* still unpaired (.rectangle == (-1, -1)) is
    paired with pos - each one's .rectangle records the other's index. A
    chosen neighbour that already has a partner is skipped, not repaired -
    once two adjacent chosen items are paired, a further chosen item showing
    up next to either of them later, to their other side, must not disturb
    that existing pairing.
    """
    item = map_of_squares[pos]
    item.state = StateEnum.chosen

    rows, cols = map_of_squares.shape
    i, j = pos
    for di, dj in DIRECT_OFFSETS:
        ni, nj = i + di, j + dj
        if not (0 <= ni < rows and 0 <= nj < cols):
            continue
        neighbour = map_of_squares[ni, nj]
        if neighbour.state == StateEnum.chosen and neighbour.rectangle == (-1, -1):
            neighbour.rectangle = pos
            item.rectangle = (ni, nj)
            break


class InvalidTilingError(RuntimeError):
    """Raised when a closure invariant over map_of_squares is violated."""
