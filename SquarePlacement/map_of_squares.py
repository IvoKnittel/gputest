from enum import Enum
from dataclasses import dataclass, field

import numpy as np

v = 0
h = 1


class StateEnum(Enum):
    free = 0
    chosen = 1
    blocked = 2


class PlacementStatus(Enum):
    candidate = 0
    unknown = 1
    eligible = 2
    unreachable = 3
    reachable = 4


@dataclass
class SquareItem:
    """One block-position's placement state.

    quality:    score used to rank candidate placements.
    state:      current placement state (free / chosen / blocked).
    alert_chosen:  raised by a neighbouring tile's placement pass instead of writing
                   .state directly; resolved into real state via place_square +
                   do_closure (see closure.do_closure's own `pos` argument).
    alert_blocked: same as alert_chosen, for the blocked outcome.
    forces1:    every (row, col) index this item forces - i.e. what must also be
                chosen if this item is chosen; empty means this item forces
                nothing (a "terminal" item, in patch terms). A set, not a list -
                membership is all that ever matters here, never order, and the
                same (row, col) index is never a meaningful duplicate of itself.
                Set by alert_graphs.set_alert_chosen_set_links: for an alert_blocked
                centre P, every one of P's own free diagonal neighbours (not P
                itself - P is the item at risk of being *blocked*, never the one
                being chosen here) gets P's corner(s) added to its own .forces1.
                A set rather than a single value because one item can legitimately
                force more than one other at once - it can be a free diagonal
                neighbour of several different alert_blocked centres at once (see
                test_complex_graphs.test_tree_fan_out), or a single centre can have
                more than one corner of its own - and none of them should be
                silently discarded in favour of another. Nothing in the current
                pipeline ever cuts this back to a subset - only clear_all_but_state
                resets it, map-wide, to empty. Where code needs just one
                representative target (e.g. to walk a single chain for display,
                or to check "does this reach a terminal") it takes an arbitrary
                entry (next(iter(...))) - stable for the duration of one pass
                since nothing mutates .forces1 mid-pass, but not any particular
                "first" one the way a list index would be.
    forced_by1: every (row, col) index that has ever forced this item - i.e. the
                other side of the same relationship .forces1 records, filled
                alongside it wherever a .forces1 entry is created
                (set_alert_chosen_set_links) so it's always available without
                needing a separate reversed copy of the map to look it up.
                Also a set, for the same reason .forces1 is.
    path_id:    ids of every group of connected alert_chosen items (linked via
                .forces1) this item belongs to - our actual building block once
                seats start linking squares together, as opposed to a lone
                square. An item can belong to more than one such group at once,
                so this is a set, not a single scalar. Seeded by
                closure.assign_paths at every "entry" (an item with .forces1 but
                no .forced_by1 - nothing causes it) and at every genuine
                diagonal-blocking pair site, then unioned forward along .forces1
                by closure.propagate_path_id_from_entries; empty set() while
                unassigned to any group. Two groups merging at some item get
                reconciled by unioning their two id sets together.
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
                propagation along .forces1 (see closure.apply_blocked_paths).
                Belongs to a separate, not-yet-wired-in alternative to
                get_blocked_links/dissolve_blocked_paths
                (closure.seed_blocked_paths/apply_blocked_paths/
                propagate_blocked_tmp_closed) - do_closure itself still runs
                the original get_blocked_links/dissolve_blocked_paths
                pipeline; nothing here is read or written by that path. Empty
                set() while unassigned.
    is_blocked_tmp: this mechanism's own marker for "just discovered
                permanently blocked by a path contradiction", kept as its own
                field so the blocked_paths mechanism never reads or writes the
                get_blocked_links/dissolve_blocked_paths pipeline's own
                bookkeeping (dissolve_blocked_paths sets no comparable flag -
                see its own docstring). Unlike dissolve_blocked_paths, nothing
                in this mechanism clears .forces1/.forced_by1/.path_id when it
                flags a cell - see apply_blocked_paths/
                propagate_blocked_tmp_closed's own docstrings. False while
                unassigned.
    forced_by:  belongs to the condition-based closure redesign (conditions.py/
                bruteforce.py/cores.py), not the alert/forces1/path_id pipeline
                above - a set of frozensets of three (id, placement_status)
                pairs ("patch ids", tagged - not plain ids, since 2026-10-22);
                if all three ids are ever chosen, this item is forced. id is a
                (row, col) index as before; placement_status is a snapshot of
                that id's own m[id].placement_status taken when the triple was
                last (re)tagged (conditions.initial_map, or
                bruteforce.retag_forced_by - see its own docstring) - like
                .forces below, this snapshot goes stale the moment
                .placement_status changes elsewhere unless something re-tags
                it; nothing does so automatically. Named to make explicit what
                it actually is: the generalization of the legacy pipeline's own
                .forced_by1 (a set of single ids, one per direct forcer) to
                n-ary forcing - "we end up with something like our alerts,
                forces, forced_by, but more general" (the original design note
                motivating this field). Seeded for the vacuum state by
                conditions.initial_map, shrunk only by discarding
                (bruteforce.compute_conditions drops a condition once one of
                its members is blocked, or consumes it - and forces this item -
                once all three are chosen); never added to afterward (the
                tagging step is a replace-in-place retag of existing triples,
                not an addition). Independent of every other field above -
                nothing in this mechanism reads or writes .forces1/.forced_by1/
                .path_id/.alert_chosen/.alert_blocked, and nothing in the
                pipeline above reads or writes .forced_by. Empty set() while
                unassigned.
    forces:     this item's own reverse index into .forced_by - the other
                direction of the same relationship, so a caller holding A
                doesn't have to scan every cell's .forced_by looking for A as
                a member. A set of (B, conditions) pairs: B is a (row, col) id
                (not tagged - B is the forced cell itself, not a triple
                member) such that A appears in at least one of B's own
                .forced_by triples, and conditions is the frozenset of exactly
                those (already id/placement_status-tagged, same objects as in
                B.forced_by) triples of B's that contain A (not all of B's
                .forced_by, only the ones A actually participates in). Seeded
                alongside .forced_by by conditions.initial_map (via bruteforce.
                update_forces), and kept in sync afterward by that same
                function - a full, not incremental, recompute - which both
                bruteforce.compute_conditions and bruteforce.compute_eligibility
                call themselves at the end of their own run, since both mutate
                .forced_by. Any other code that mutates .forced_by directly
                needs to call update_forces(m) itself afterward, or this field
                goes stale relative to it. Empty set() while unassigned.
    placement_status: a PlacementStatus value - renamed from the earlier plain
                True/False/None `.eligible` field (same underlying idea,
                generalized to five states instead of three booleans/None).
                `unknown` is the default/reset state. `candidate` marks a cell
                inside one of the fixed "placement" cores (see
                test_context_tiles.py - the cores at index (1,1), (1+4I,1+4J)
                for I,J=0,... of the current core grid; every other, "transit"
                core's cells stay `unknown`) - set directly at setup, not
                computed. bruteforce.compute_eligibility(m, realm) still
                overwrites this for every cell in m, not just the ones in
                realm, the same way it overwrote .eligible before the rename:
                a cell outside realm -> `eligible` (was True) - we track
                nothing about it from this realm's own vantage point, so
                there is no evidence to rule it out. A cell inside realm
                starts `unknown` (was None), and is only set `eligible` (was
                True) once compute_eligibility finds one of its own
                .forced_by all of whose members are already `eligible` -
                iterated to a fixed point, so eligibility can chain through
                other inside cells, not just reach straight to the realm
                boundary. `unreachable` and `reachable` are reserved, not yet
                produced by any code - `unreachable` is the enum's analog of
                the old unused False outcome; `reachable` is new, meaning not
                yet assigned by anything.
    """
    quality: float = -1.0
    state: StateEnum = StateEnum.free
    alert_chosen: bool = False
    alert_blocked: bool = False
    forces1: set = field(default_factory=set)
    forced_by1: set = field(default_factory=set)
    path_id: set = field(default_factory=set)
    rectangle: tuple = (-1, -1)
    blocked_paths: set = field(default_factory=set)
    forced_by: set = field(default_factory=set)
    forces: set = field(default_factory=set)
    is_blocked_tmp: bool = False
    placement_status: object = PlacementStatus.unknown
         
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
