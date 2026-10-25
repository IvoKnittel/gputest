"""Map construction for the condition-based closure redesign:
vacuum_condition_offsets, initial_map - the vacuum state the algorithm itself
(bruteforce.py) runs on, plus the helpers that build it. See cores.py for the
patch-id/alias side; see bruteforce.py for the actual closure algorithm -
deliberately kept out of this file, which is map/data-structure construction
only.

The base-level map is an array of real map_of_squares.SquareItem - see its
own .forced_by field docstring - not a separate prototype class; this
mechanism is independent of (reads/writes none of) SquareItem's existing
alert/forces1/forced_by1/path_id fields, so the two can share one object
safely.

A condition is a frozenset of three patch ids: if all three are placed
(chosen), the cell must be placed too. Each one describes a "seat" - a 2x2
block whose three other corners are all blocked while the cell itself is
still free - and the three ids are the placements that produce those three
blocked corners (a corner is blocked by a chosen diagonal neighbour).
"""
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from map_of_squares import SquareItem
from bruteforce import init_vacuum

_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SEAT_CORNERS = ((-1, -1), (-1, 0), (0, -1))


def _is_diagonal(a, b):
    return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1


def _rotate(offset):
    return (offset[1], -offset[0])


@lru_cache(maxsize=None)
def vacuum_condition_offsets():
    """Every distinct set of three placement offsets, relative to a cell x at
    (0, 0), that leaves x free but blocks all three other corners of one of
    its four surrounding 2x2 blocks (76 in total).

    Built for the seat with corners (-1,-1), (-1,0), (0,-1): each corner is
    blocked by a chosen diagonal neighbour of it, excluding cells that can't be
    chosen (x, the seat's own corners, and x's own diagonal neighbours, which
    would block x). One blocker per corner, and no two of them diagonal to each
    other (a diagonal-chosen pair is invalid). The other three seats are the
    90-degree rotations of that one; triples that serve two seats coincide.

    Cached (lru_cache, no arguments so effectively computed once) - this is
    called once per cell by bruteforce.init_vacuum, so an uncached board
    build recomputes the same 76-triple set thousands of times over. Safe to
    cache and hand back the same object every time: every caller only ever
    iterates this set, nothing mutates it.
    """
    forbidden = set(_SEAT_CORNERS) | {(0, 0)} | {(-1, 1), (1, -1), (1, 1)}
    per_corner = [[(c[0] + d[0], c[1] + d[1]) for d in _DIAGONALS
                   if (c[0] + d[0], c[1] + d[1]) not in forbidden]
                  for c in _SEAT_CORNERS]
    seat = set()
    for blockers in product(*per_corner):
        if any(_is_diagonal(a, b) for a, b in combinations(blockers, 2)):
            continue
        seat.add(frozenset(blockers))

    offsets = set()
    for _ in range(4):
        offsets |= seat
        seat = {frozenset(_rotate(p) for p in triple) for triple in seat}
    return offsets


def initial_map(shape):
    """The vacuum state on a (rows, cols) board: an object array of
    map_of_squares.SquareItem, all free (every other field at its own
    default), each with .forced_by = the vacuum triples whose three
    placements all lie on the board, and .forces = the reverse index built
    from that same data - see SquareItem.forces's own docstring. A triple's
    three members are (id, placement_status) pairs, not plain ids - id is a
    (row, col) index as before, placement_status is a snapshot of that id's
    own m[id].placement_status at the moment this function tags it (see
    SquareItem.placement_status's own docstring). Every cell starts at this
    function's own default PlacementStatus.unknown (nothing has called
    test_context_tiles.mark_placement_cores or compute_eligibility yet at
    board-build time), so every tag here is unknown - the tags only become
    informative once something sets real .placement_status values and then
    calls bruteforce.retag_forced_by(m) to refresh them; this function does
    not do that refreshing itself, only the initial (uniformly unknown) tag.

    Built from bruteforce.init_vacuum(id_), which computes both fields for
    one id in isolation but knows nothing about board edges (see its own
    docstring) - this function supplies the one thing init_vacuum can't: for
    every (row, col) id, keep only the on-board positions, then bounds-filter
    what init_vacuum returns around that id.

    A triple with an off-board placement can never be completed, so it is
    left out entirely - not shrunk - since an off-board blocker candidate is
    just one of several alternative combinations for that corner (see
    vacuum_condition_offsets - each corner has up to 3 candidate blockers,
    each giving its own triple); the corner can still be blocked via a
    different, on-board combination, already present as a separate triple.
    Dropping only the off-board member rather than the whole triple was tried
    and is wrong - it manufactures a forced cell out of a combination that
    was never really available, and does not match how off-board is treated
    anywhere else in this codebase (find_secondary_links's _OFFBOARD sentinel
    is inert - neither free nor blocked - not a stand-in for blocked). Cells
    near the edge simply have fewer conditions than an interior cell's 76.

    .forces is bounds-filtered the same way, one level deeper: a (B,
    conditions) pair from init_vacuum is kept only if B itself is on-board,
    and even then only with conditions narrowed to the ones that are
    themselves fully on-board (same triple rule as above) - a pair whose
    every condition gets filtered away this way is dropped entirely, there's
    nothing left for it to mean. Confirmed equivalent to the older, two-pass
    construction (build every on-board .forced_by directly, then invert the
    whole board via bruteforce.update_forces) on the same board - identical
    per-cell .forced_by and .forces, verified over an 11x11 board.

    Fast path: init_vacuum((i, j)) never reaches further than 2 cells from
    (i, j) for .forced_by's own members or for a .forces entry's own B - and
    never further than 3 for the members *inside* a .forces entry's own
    conditions (one more hop, through B's own triple) - both confirmed
    empirically, not assumed. So for a cell at least 2 cells from every edge,
    .forced_by and every .forces entry's B are already guaranteed on-board -
    no filtering needed there; only the nested conditions inside .forces
    still need checking, unless the cell is at least 3 cells from every edge,
    in which case nothing needs checking at all. Using margin 2 for the
    nested check too would be wrong - a cell exactly 2 cells from an edge can
    have a nested member one cell past it. Re-verified byte-for-byte against
    the unoptimized version (every branch below gives the same .forced_by/
    .forces as filtering everything unconditionally) over a 15x15 board.
    """
    rows, cols = shape
    FORCED_BY_REACH = 2
    FORCES_CONDITION_REACH = 3

    def on_board(id_):
        r, c = id_
        return 0 <= r < rows and 0 <= c < cols

    def clear_of_every_edge(i, j, margin):
        return margin <= i < rows - margin and margin <= j < cols - margin

    m = np.empty(shape, dtype=object)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = SquareItem()

    def tag_condition(condition):
        return frozenset((member, m[member].placement_status) for member in condition)

    for i in range(rows):
        for j in range(cols):
            forced_by, forces = init_vacuum((i, j))
            b_reach_clear = clear_of_every_edge(i, j, FORCED_BY_REACH)
            nested_reach_clear = clear_of_every_edge(i, j, FORCES_CONDITION_REACH)

            if b_reach_clear:
                m[i, j].forced_by = {tag_condition(condition) for condition in forced_by}
            else:
                m[i, j].forced_by = {tag_condition(condition) for condition in forced_by
                                      if all(on_board(member) for member in condition)}

            if nested_reach_clear:
                m[i, j].forces = {(b, frozenset(tag_condition(condition) for condition in conditions))
                                   for b, conditions in forces}
            else:
                kept_forces = set()
                for b, conditions in forces:
                    if not b_reach_clear and not on_board(b):
                        continue
                    kept_conditions = frozenset(tag_condition(condition) for condition in conditions
                                                 if all(on_board(member) for member in condition))
                    if kept_conditions:
                        kept_forces.add((b, kept_conditions))
                m[i, j].forces = kept_forces

    return m
