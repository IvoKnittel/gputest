from enum import Enum
from dataclasses import dataclass

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
    graph_id:      id of the connected group of linked alert_chosen items this one
                    belongs to, assigned by do_closure; -1 while unassigned. How large
                    these groups can grow is an open question - see the "Open
                    question" note in do_closure's docstring and test_graph_spread.
    """
    quality: float = -1.0
    state: StateEnum = StateEnum.free
    alert_chosen: bool = False
    alert_blocked: bool = False
    link: "tuple[int, int] | None" = None
    graph_id: int = -1


def is_free(map_of_squares, r, c):
    """A cell can be chosen only while it is still in its initial, untouched state."""
    return map_of_squares[r, c].state == StateEnum.free


def place_square_in_core(map_of_squares, core_origin, sz_core):
    """Simulate one CUDA call: place the best-quality square in the 3x3 core of a tile.

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


def get_placement_map(map_of_squares):
    """Collapse each cell's .state into a display value: chosen=1, blocked=-1, free=0."""
    rows, cols = map_of_squares.shape
    display = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            state = map_of_squares[i, j].state
            if state == StateEnum.chosen:
                display[i, j] = 1.0
            elif state == StateEnum.blocked:
                display[i, j] = -1.0
    return display


class InvalidTilingError(RuntimeError):
    """Raised when a closure invariant over map_of_squares is violated."""
