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
    blocked_tmp: True on a cell closure.set_blocked_links has flagged as the
                origin of a self-contradicting path. Not a distinct state -
                .state is already the real StateEnum.blocked the moment this
                is set (see set_blocked_links) - just a marker so display can
                colour a genuine-but-just-discovered block differently from
                one that was always blocked. Cleared back to False by
                closure.clear_all_but_state, along with everything else.
    alert_chosen:  raised by a neighbouring tile's placement pass instead of writing
                   .state directly; resolved into real state via forced_closure +
                   place_squares (see test_utils.place_and_chase).
    alert_blocked: same as alert_chosen, for the blocked outcome.
    forces:     every (row, col) index this item forces - i.e. what must also be
                chosen if this item is chosen; empty means this item forces
                nothing (a "terminal" item, in patch terms). A set, not a list -
                membership is all that ever matters here, never order, and the
                same (row, col) index is never a meaningful duplicate of itself.
                Set by alert_graphs.set_alert_chosen_set_links: for an alert_blocked
                centre P, every one of P's own free diagonal neighbours (not P
                itself - P is the item at risk of being *blocked*, never the one
                being chosen here) gets P's corner(s) added to its own .forces.
                A set rather than a single value because one item can legitimately
                force more than one other at once - it can be a free diagonal
                neighbour of several different alert_blocked centres at once (see
                test_complex_graphs.test_tree_fan_out), or a single centre can have
                more than one corner of its own - and none of them should be
                silently discarded in favour of another. set_blocked_links is
                the only place this gets cut back to set() now, when setting a
                cell's .blocked_tmp flag, severing the pairing
                entirely. Where code needs just one representative target (e.g. to
                walk a single chain for display, or to check "does this reach a
                terminal") it takes an arbitrary entry (next(iter(...))) - stable for
                the duration of one pass since nothing mutates .forces mid-pass, but
                not any particular "first" one the way a list index would be.
    forced_by:  every (row, col) index that has ever forced this item - i.e. the
                other side of the same relationship .forces records, filled
                alongside it wherever a .forces entry is created
                (set_alert_chosen_set_links) so it's always available without
                needing a separate reversed copy of the map to look it up.
                Also a set, for the same reason .forces is. set_blocked_links
                retracts both directions when it cuts .forces: the cut cell's
                own .forced_by is cleared too, and it's removed from every one
                of its former forcers' .forces sets, not just the targets it
                used to force.
    centrality: distance, in .forces hops, from this item's chain to the closest
                terminal (no-.forces) item of its patch - part of a separate
                centrality/path_id mechanism from the one path_id (below)
                actually uses; currently unused, since find_central_patch_items,
                its only writer, has been removed. -1 while unassigned.
    path_id:    ids of every group of connected alert_chosen items (linked via
                .forces) this item belongs to - our actual building block once
                seats start linking squares together, as opposed to a lone
                square. An item can belong to more than one such group at once,
                so this is a set, not a single scalar. Seeded by
                closure.assign_paths at every "entry" (an item with .forces but
                no .forced_by - nothing causes it) and at every genuine
                diagonal-blocking pair site, then unioned forward along .forces
                by closure.propagate_path_id_from_entries; empty set() while
                unassigned to any group. Two groups merging at some item get
                reconciled by unioning their two id sets together.
    max_id:     candidate for the largest flattened index among a pure ring's
                members - part of the same now-removed centrality/path_id
                mechanism as .centrality (find_cycle_patches was its only
                reader/writer, and has itself been removed); currently unused.
                -1 while unassigned.
    rectangle:  (row, col) index of the direct neighbour this item has been
                paired with into a domino, or (-1, -1) while unpaired. Set by
                set_square_chosen: when an item is chosen, the first direct
                neighbour found that is itself already chosen and still
                unpaired becomes its partner, both ways. Deliberately
                first-come: a chosen neighbour that already has a partner is
                skipped rather than repaired, so pairing one more chosen item
                onto either end of an existing domino later never disturbs it.
    blocked_paths: ids (a subset of path_id) this item has learned are
                self-contradicting - the same fact get_blocked_links computes
                into its own returned set, but kept per-cell here instead of
                collapsed into one global id set, plus one hop of forward
                propagation along .forces (see closure.apply_blocked_paths).
                Belongs to a separate, not-yet-wired-in alternative to
                get_blocked_links/set_blocked_links (closure.seed_blocked_paths/
                apply_blocked_paths/propagate_blocked_tmp_closed) - do_closure
                itself still runs the original get_blocked_links/set_blocked_links
                pipeline; nothing here is read or written by that path. Empty
                set() while unassigned.
    is_blocked_tmp: this mechanism's own marker for "just discovered
                permanently blocked by a path contradiction" - the counterpart
                to blocked_tmp above, but for the blocked_paths mechanism, kept
                as a distinct field so the two mechanisms never read or write
                each other's bookkeeping. Unlike set_blocked_links, nothing in
                this mechanism clears .forces/.forced_by/.path_id when it flags
                a cell - see apply_blocked_paths/propagate_blocked_tmp_closed's
                own docstrings. False while unassigned.
    """
    quality: float = -1.0
    state: StateEnum = StateEnum.free
    blocked_tmp: bool = False
    alert_chosen: bool = False
    alert_blocked: bool = False
    forces: set = field(default_factory=set)
    forced_by: set = field(default_factory=set)
    centrality: int = -1
    path_id: set = field(default_factory=set)
    max_id: int = -1
    rectangle: tuple = (-1, -1)
    blocked_paths: set = field(default_factory=set)
    is_blocked_tmp: bool = False
         
# The four direct (orthogonal) neighbours - as opposed to a diagonal one -
# checked by set_square_chosen when pairing a newly-chosen item into a domino.
DIRECT_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def set_square_chosen(map_of_squares, pos):
    """Wrapper around setting map_of_squares[pos].state to StateEnum.chosen,
    that also pairs it into a rectangle (domino) with a direct neighbour, if
    one qualifies. Lives here, not in closure.py/representation.py, so both
    (representation.py can't import from closure.py without a cycle) can call
    it as the one place .state ever becomes StateEnum.chosen.

    Raises InvalidTilingError if pos isn't currently StateEnum.free - reachable,
    not just defensive: placing a square on a cell closure has already
    legitimately blocked (as a diagonal neighbour of an earlier placement)
    would, without this check, silently overwrite the block back to chosen,
    producing an invalid diagonal-chosen pair only caught later, incidentally,
    by real_space_map. Rejecting it right here surfaces the actual mistake
    (choosing an illegal cell) at the point it happens, instead of some
    unrelated-looking failure downstream.

    Checks all four direct neighbours of pos: the first one found that is
    itself already chosen *and* still unpaired (.rectangle == (-1, -1)) is
    paired with pos - each one's .rectangle records the other's index. A
    chosen neighbour that already has a partner is skipped, not repaired -
    once two adjacent chosen items are paired, a further chosen item showing
    up next to either of them later, to their other side, must not disturb
    that existing pairing.
    """
    item = map_of_squares[pos]
    if item.state != StateEnum.free:
        raise InvalidTilingError(f"cannot choose {pos}: state is {item.state}, not free")
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
