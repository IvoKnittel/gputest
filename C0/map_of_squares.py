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
                   .state directly; resolved into real state by do_closure().
    alert_blocked: same as alert_chosen, for the blocked outcome.
    forces:     every (row, col) index this item forces - i.e. what must also be
                chosen if this item is chosen - in the order each link was made;
                empty means this item forces nothing (a "terminal" item, in patch
                terms). Set by alert_graphs.set_alert_chosen: for an alert_blocked
                centre P, every one of P's own free diagonal neighbours (not P
                itself - P is the item at risk of being *blocked*, never the one
                being chosen here) gets P's corner(s) appended to its own .forces.
                A list rather than a single value because one item can legitimately
                force more than one other at once - it can be a free diagonal
                neighbour of several different alert_blocked centres at once (see
                test_complex_graphs.test_tree_fan_out), or a single centre can have
                more than one corner of its own - and none of them should be
                silently discarded in favour of another. A cut (find_central_patch_items,
                find_cycle_patches) clears this back to [], severing the pairing
                entirely.
    forced_by:  every (row, col) index that has ever forced this item - i.e. the
                other side of the same relationship .forces records, filled
                alongside it wherever a .forces entry is created (set_alert_chosen,
                copy_map_reverse, representation.set_link) so it's always available
                without needing a separate reversed copy of the map to look it up.
                remove_blocked_links retracts an entry here when it prunes the
                matching .forces link it turns out to be doomed, so a .forced_by
                entry stays live rather than describing a forcing relationship
                that no longer holds. A .forces cut (find_central_patch_items,
                find_cycle_patches) does not retract, though - a stale entry can
                still be left behind that way.
    conflicts:  patch_ids of other patches that exclude this item's patch - i.e.
                choosing a member of this patch would block a member of theirs.
                Populated by do_closure once patch_id has settled; empty until
                then.
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
                .forces[0] at once (see .forces above and test_do_closure_steps)
                can still have one candidate crowd out another here - see the
                "Known gap" note on find_cycle_patches.
    """
    quality: float = -1.0
    state: StateEnum = StateEnum.free
    alert_chosen: bool = False
    alert_blocked: bool = False
    forces: list = field(default_factory=list)
    forced_by: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    centrality: int = -1
    patch_id: int = -1
    max_id: int = -1


@dataclass
class CoreItem:
    """Summary of one 3x3 tile-core, for exchanging conflict information between
    cores rather than between individual items (see closure.build_core_map).

    conflicts: patch_ids that conflict with some patch represented in this core -
               the union of every item's .conflicts within the core, with any
               patch_id that is *also* one of this core's own items' .patch_id
               discarded: an intra-core conflict can never actually matter, since
               only one patch is ever chosen per core in the first place. Built by
               closure.build_core_map; see closure.map_patches_to_pivots for the
               matching per-patch side of this scheme.
    """
    conflicts: list = field(default_factory=list)

class InvalidTilingError(RuntimeError):
    """Raised when a closure invariant over map_of_squares is violated."""
