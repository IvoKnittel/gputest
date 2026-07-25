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

    quality:       score used to rank candidate placements (from image_squares_ranked).
    state:         current placement state (free / chosen / blocked).
    alert_chosen:  raised by a neighbouring tile's placement pass instead of writing
                   .state directly; resolved into real state by do_closure().
    alert_blocked: same as alert_chosen, for the blocked outcome.
    link:          (row, col) index of another item in map_of_squares this one is
                    paired with - e.g. an alert_blocked item points at the
                    alert_chosen item that would resolve its risk - or None.
    conflicts:     patch_ids of other patches that exclude this item's patch - i.e.
                    choosing a member of this patch would block a member of theirs.
                    Populated by do_closure once patch_id has settled; empty until
                    then.
    centrality:    distance, in .link hops, from this item's chain to the closest
                    terminal (no-.link) item of its patch; assigned by
                    closure.find_central_patch_items one generation at a time, -1
                    while unassigned.
    patch_id:      id of the patch this item belongs to. A "patch" is a connected
                    group of alert_chosen items linked together - our actual
                    building block once alerts start linking squares together, as
                    opposed to a lone square. Seeded at a terminal (its own
                    flattened index) and propagated outward one generation at a
                    time by closure.find_central_patch_items; -1 while unassigned.
                    Two patches merging get reconciled to whichever id is larger;
                    a patch looping back on its own id instead has its closing
                    link cut, so cycles don't propagate forever. (This field used
                    to be called path_id, back when we still called this concept
                    a "path" rather than a "patch".)
    max_id:        candidate for the largest flattened index among a pure ring's
                    members (one with no terminal anywhere, so centrality/patch_id
                    never reach it) - used by closure.find_cycle_patches as a
                    parallel-safe way to agree on a single break point without
                    relying on processing order; -1 while unassigned, and reset
                    back to -1 as soon as this item (or its linked item) gets a real
                    centrality, since max_id has no meaning outside a pure ring.
    """
    quality: float = -1.0
    state: StateEnum = StateEnum.free
    alert_chosen: bool = False
    alert_blocked: bool = False
    link: "tuple[int, int] | None" = None
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


def is_free(map_of_squares, r, c):
    """A cell can be chosen only while it is still in its initial, untouched state."""
    return map_of_squares[r, c].state == StateEnum.free


def place_square_in_core(map_of_squares, core_origin, sz_core):
    """Simulate one CUDA call: place the best-quality square in the 3x3 core of a tile.

    This is single-square placement, the low-level mechanical primitive - not to be
    confused with "patch placement" (see closure.py), the higher-level operation we
    actually build the plane out of: a patch is a connected group of alert_chosen
    squares that must be placed together, and only ever consists of a single square
    for the first couple of sublattice passes, before any alerts have had a chance
    to link squares together at all.

    A call commits state directly, both inside its own core and on the four diagonal
    neighbours of the chosen cell, which can fall in the 1-cell border shared with (or
    even inside the core of) a neighbouring tile. This is only safe because the caller
    only ever activates one sublattice of tiles at a time (see `shifts` in
    test_square_placement): simultaneously active tiles are 2 tiles apart, so no other
    active tile's core or border is ever touched by this write in the same pass.

    map_of_squares: full padded map of squares (read-only in the border, writable core)
                    .quality
                    .state
    core_origin:    (row, col) of the top-left of the 3x3 core in padded coordinates
    sz_core:        size of the mutable core (3)
    """

    # Find the highest-quality unoccupied position in the core.
    best_quality = -1.0
    best_pos = None
    for di in range(sz_core):
        for dj in range(sz_core):
            r, c = core_origin[v] + di, core_origin[h] + dj
            if is_free(map_of_squares, r, c) and map_of_squares[r, c].quality > best_quality:
                best_quality = map_of_squares[r, c].quality
                best_pos = (r, c)

    if best_pos is not None:
        map_of_squares[best_pos[v], best_pos[h]].state = StateEnum.chosen
        map_of_squares[best_pos[v]-1, best_pos[h]-1].state = StateEnum.blocked
        map_of_squares[best_pos[v]-1, best_pos[h]+1].state = StateEnum.blocked
        map_of_squares[best_pos[v]+1, best_pos[h]-1].state = StateEnum.blocked
        map_of_squares[best_pos[v]+1, best_pos[h]+1].state = StateEnum.blocked

def map_of_squares_from_quality(map_of_squares, quality_padded):
    """Fill every cell of map_of_squares with a SquareItem built from quality_padded.

    Padding cells (quality < 0) start out blocked so they can never be selected as a
    placement, matching the -1 sentinel used throughout image_to_squares.py.
    """
    rows, cols = quality_padded.shape
    for i in range(rows):
        for j in range(cols):
            quality = quality_padded[i, j]
            state = StateEnum.blocked if quality < 0 else StateEnum.free
            map_of_squares[i, j] = SquareItem(quality=quality, state=state)




class InvalidTilingError(RuntimeError):
    """Raised when a closure invariant over map_of_squares is violated."""
