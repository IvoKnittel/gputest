"""Vacuum state for the condition-based closure redesign (see cores.py for the
patch-id/alias side). Nothing is placed anywhere; every cell's patch id is its
own (row, col) index, and its .conditions lists every way the cell could
become forced.

A condition is a frozenset of three patch ids: if all three are placed
(chosen), the cell must be placed too. Each one describes a "seat" - a 2x2
block whose three other corners are all blocked while the cell itself is
still free - and the three ids are the placements that produce those three
blocked corners (a corner is blocked by a chosen diagonal neighbour).
"""
from itertools import combinations, product

import numpy as np

from map_of_squares import StateEnum

_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SEAT_CORNERS = ((-1, -1), (-1, 0), (0, -1))


class ConditionItem:
    """One cell: .state, .patch_id (initially the cell's own (row, col)) and
    .conditions, a set of frozensets of three patch ids - see module docstring.
    """
    def __init__(self, patch_id):
        self.state = StateEnum.free
        self.patch_id = patch_id
        self.conditions = set()


def _is_diagonal(a, b):
    return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1


def _rotate(offset):
    return (offset[1], -offset[0])


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


def compute_conditions(m):
    """Drop every condition whose outcome is now decided, in place.

    An element id is looked up as a cell index: m[id].state. A condition is
    false, and removed, if any of its elements is blocked - that placement can
    never happen. It is true, and removed, if all of its elements are chosen -
    its cell is then forced. Conditions with no blocked element and at least
    one element still free are kept unchanged.

    A cell that is already chosen or blocked has nothing left to be forced
    into, so all of its conditions are cleared.

    Returns the list of (row, col) positions of free cells that had a
    condition become true; placing them is left to the caller.
    """
    forced = []
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if item.state != StateEnum.free:
                item.conditions.clear()
                continue
            became_true = False
            kept = set()
            for condition in item.conditions:
                states = [m[pos].state for pos in condition]
                if StateEnum.blocked in states:
                    continue
                if all(s == StateEnum.chosen for s in states):
                    became_true = True
                    continue
                kept.add(condition)
            item.conditions = kept
            if became_true:
                forced.append((i, j))
    return forced


def initial_map(shape):
    """The vacuum state on a (rows, cols) board: an object array of
    ConditionItem, all free, each with patch_id = its own (row, col) and
    .conditions = the vacuum triples (ids as (row, col)) whose three
    placements all lie on the board. A triple with an off-board placement can
    never be completed, so it is left out; cells near the edge simply have
    fewer conditions.
    """
    rows, cols = shape
    offsets = vacuum_condition_offsets()
    m = np.empty(shape, dtype=object)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = ConditionItem((i, j))
    for i in range(rows):
        for j in range(cols):
            for triple in offsets:
                ids = frozenset((i + di, j + dj) for di, dj in triple)
                if all(0 <= r < rows and 0 <= c < cols for r, c in ids):
                    m[i, j].conditions.add(ids)
    return m
