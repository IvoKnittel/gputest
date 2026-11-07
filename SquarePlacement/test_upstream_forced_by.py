"""Tests for the "last cell inside" convention of CoreItem.forced_by /
.placement_reach, and for composing a parent's lists from a 3x3 array of
child CoreItems on the way up (2026-11-07).

Convention (Ivo, 2026-11-07): a pair (id_in, id_out) in a tile's
.forced_by means a chain of single links enters the tile from id_in and
leaves it at id_out, where
- id_in is the cell OUTSIDE the tile that links into it, or a candidate
  cell inside the tile (a chain starting at a placement decision);
- id_out is the LAST cell INSIDE the tile that has a link pointing out.

With this convention one tile's id_out is exactly the next tile's id_in,
so a parent can join child pairs across a border. Implemented 2026-11-07
in cores.py (CoreItem._reach, .exits, _compose_up); before that, id_out was
the first cell OUTSIDE the tile and every chain was cut at the first border
between two children - see upstream-forced-by-current.html (project root)
for what the old code produced on the same scenario.

The scenario test reproduces upstream-forced-by.html (project root): a
parent P over 3x3 child tiles of 3x3 cells each, tile (1,1) the placement
tile with candidate (4,4), and seven link scenarios. Coordinates below are
the drawing's own (-1..9, P itself is 0..8); `at` shifts them by one into
array indices, since the drawing's outside ring at -1 needs a real cell.
"""
import numpy as np
from bruteforce import update_forces
from cores import CoreItem
from map_of_squares import SquareItem, PlacementStatus

def at(r, c):
    """Drawing coordinates -> array index (the drawing starts at -1)."""
    return (r + 1, c + 1)


def pairs(*drawing_pairs):
    """{((r,c),(r,c)), ...} in drawing coordinates -> array indices."""
    return {(at(*a), at(*b)) for a, b in drawing_pairs}


def build_map(shape, chains, candidates=()):
    """A map of SquareItems whose only conditions are the single links
    along `chains` (each a list of drawing coordinates, forcer -> forced).
    A cell reached by two chains (fan-in) gets two size-1 conditions.
    .forces is rebuilt from .forced_by via update_forces."""
    m = np.empty(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m[i, j] = SquareItem()
    cand = {at(*c) for c in candidates}
    for idx in cand:
        m[idx].placement_status = PlacementStatus.candidate
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            a_idx, b_idx = at(*a), at(*b)
            status = PlacementStatus.candidate if a_idx in cand else PlacementStatus.unknown
            m[b_idx].forced_by.add(frozenset({(a_idx, status)}))
    update_forces(m)
    return m


def child_tile(m, i, j):
    """Child tile (i, j) of P: drawing rows 3i..3i+2, cols 3j..3j+2."""
    r0, c0 = at(3 * i, 3 * j)
    return CoreItem(m[r0:r0 + 3, c0:c0 + 3], margin=None, realm=((r0, r0 + 3), (c0, c0 + 3)))


# Scenarios of upstream-forced-by.html, in drawing coordinates.
CHAINS = [
    # A: path through several tiles (0,0) -> (0,1) -> (0,2), leaves P
    [(1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)],
    # B: enters and leaves the same tile (2,0)
    [(9, 1), (8, 1), (7, 1), (7, 0), (7, -1)],
    # C: fan-out from candidate (4,4); one branch leaves P via (1,2),
    #    the other dies at (2,4) inside tile (0,1)
    [(4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9)],
    [(4, 4), (3, 4), (2, 4)],
    # D: fan-in at (7,7) in tile (2,2), continues through (1,2), leaves P
    [(9, 7), (8, 7), (7, 7)],
    [(7, 9), (7, 8), (7, 7)],
    [(7, 7), (6, 7), (5, 7), (5, 8), (5, 9)],
    # E: entry from above P at (-1,5), through tile (0,1) into (0,2), leaves P
    [(-1, 5), (0, 5), (0, 6), (0, 7), (0, 8), (-1, 8)],
    # F: leaves tile (0,0) and dies at (2,3) on entering tile (0,1)
    [(2, -1), (2, 0), (2, 1), (2, 2), (2, 3)],
    # G: dangling start (8,2) in tile (2,0), runs through tile (2,1)
    [(8, 2), (8, 3), (8, 4), (8, 5), (9, 5)],
]
CANDIDATES = [(4, 4)]

EXPECTED_CHILD_FORCED_BY = {
    (0, 0): pairs(((1, -1), (1, 2)), ((2, -1), (2, 2))),
    (0, 1): pairs(((1, 2), (1, 5)), ((-1, 5), (0, 5))),
    (0, 2): pairs(((1, 5), (1, 8)), ((0, 5), (0, 8))),
    (1, 0): set(),
    (1, 1): pairs(((4, 4), (4, 5)), ((4, 4), (3, 4))),
    (1, 2): pairs(((4, 5), (4, 8)), ((6, 7), (5, 8))),
    (2, 0): pairs(((9, 1), (7, 0))),
    (2, 1): pairs(((8, 2), (8, 5))),
    (2, 2): pairs(((9, 7), (6, 7)), ((7, 9), (6, 7))),
}
EXPECTED_CHILD_PLACEMENT_REACH = {
    tile: (pairs(((4, 4), (4, 5)), ((4, 4), (3, 4))) if tile == (1, 1) else set())
    for tile in EXPECTED_CHILD_FORCED_BY
}
EXPECTED_PARENT_FORCED_BY = pairs(
    ((1, -1), (1, 8)),   # A
    ((9, 1), (7, 0)),    # B
    ((4, 4), (4, 8)),    # C, the branch that leaves P
    ((9, 7), (5, 8)),    # D
    ((7, 9), (5, 8)),    # D
    ((-1, 5), (0, 8)),   # E
)
EXPECTED_PARENT_PLACEMENT_REACH = pairs(((4, 4), (4, 8)))


def _children(m):
    kids = {}
    for i in range(3):
        for j in range(3):
            k = child_tile(m, i, j)
            k.update_placement_reach()
            k.update_forced_by()
            kids[i, j] = k
    return kids


def test_three_tile_straight_chain_joins_across_child_borders():
    """Smallest case: one straight chain of single links along row 1, from
    (1,2) outside P through three child tiles (cols 3-5, 6-8, 9-11) to
    (1,12) outside P. Each child's id_out is its last cell, which is the
    next child's id_in, so the parent gets one pair, ending at (1,11), the
    last cell inside P. Uses array coordinates directly (no `at` shift)."""
    m = np.empty((3, 15), dtype=object)
    for i in range(3):
        for j in range(15):
            m[i, j] = SquareItem()
    for j in range(3, 13):
        m[1, j].forced_by = {frozenset({((1, j - 1), PlacementStatus.unknown)})}
    update_forces(m)

    kids = [CoreItem(m[0:3, c:c + 3], margin=None, realm=((0, 3), (c, c + 3))) for c in (3, 6, 9)]
    for k in kids:
        k.update_placement_reach()
        k.update_forced_by()
    assert [k.forced_by for k in kids] == [
        {((1, 2), (1, 5))},
        {((1, 5), (1, 8))},
        {((1, 8), (1, 11))},
    ]

    parent = CoreItem(np.array([kids], dtype=object), margin=None, realm=((0, 3), (3, 12)))
    parent.update_placement_reach()
    parent.update_forced_by()
    assert parent.forced_by == {((1, 2), (1, 11))}
    assert parent.placement_reach == set()


def test_child_lists_use_last_cell_inside():
    """Every child tile of the drawing: .forced_by and .placement_reach
    exactly as listed in upstream-forced-by.html. Includes the three pairs
    that will NOT join at the parent: ((4,4),(3,4)) and ((2,-1),(2,2))
    (id_out with no matching id_in - the chain dies in the next tile), and
    ((8,2),(8,5)) (id_in inside P, not a candidate, and no child ends
    there)."""
    m = build_map((11, 11), CHAINS, CANDIDATES)
    kids = _children(m)
    assert {t: k.forced_by for t, k in kids.items()} == EXPECTED_CHILD_FORCED_BY
    assert {t: k.placement_reach for t, k in kids.items()} == EXPECTED_CHILD_PLACEMENT_REACH


def test_parent_joins_matching_child_pairs_and_drops_the_rest():
    """Parent P over the 3x3 children: matching child pairs join into one
    parent pair (A through three tiles, C's surviving branch, both D
    fan-in chains, E entering from above); B passes through a single
    tile; the dead ends (C's other branch, F) and the dangling start (G)
    contribute nothing."""
    m = build_map((11, 11), CHAINS, CANDIDATES)
    kids = _children(m)
    elements = np.empty((3, 3), dtype=object)
    for (i, j), k in kids.items():
        elements[i, j] = k
    r0, c0 = at(0, 0)
    parent = CoreItem(elements, margin=None, realm=((r0, r0 + 9), (c0, c0 + 9)))
    parent.update_placement_reach()
    parent.update_forced_by()
    assert parent.forced_by == EXPECTED_PARENT_FORCED_BY
    assert parent.placement_reach == EXPECTED_PARENT_PLACEMENT_REACH
