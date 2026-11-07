"""Tests for the "last cell inside" convention of CoreItem.links (named
.forced_by until 2026-11-07, now (id_in, id_out, id_exit) triples), and for
composing a parent's lists from a 3x3 array of child CoreItems on the way
up (2026-11-07).

Convention (Ivo, 2026-11-07): a triple (id_in, id_out, id_exit) in a
tile's .links means a chain of single links enters the tile from id_in and
leaves it at id_out, where
- id_in is the cell OUTSIDE the tile that links into it, or a candidate
  cell inside the tile (a chain starting at a placement decision);
- id_out is the LAST cell INSIDE the tile that has a link pointing out;
- id_exit is the cell outside the tile that link points to.

With this convention one tile's id_out is exactly the next tile's id_in,
so a parent can join child pairs across a border. Implemented 2026-11-07
in cores.py (Base.reach, Upstream.compose_up); before that, id_out was
the first cell OUTSIDE the tile and every chain was cut at the first border
between two children - see upstream-forced-by-current.html (project root)
for what the old code produced on the same scenario.

Candidates are whole placement tiles (Pyramid.is_candidate, global layout;
.placement_reach was removed 2026-11-07 - Pyramid.candidate_links() is the
candidate-started part of .links). In the scenario, tile (1,1) is the
placement tile, so every one of its cells is a candidate - not just (4,4),
the start of chain C, but also (4,5) and (3,4) further along that chain.

The scenario test reproduces upstream-forced-by.html (project root): a
parent P over 3x3 child tiles of 3x3 cells each and seven link scenarios.
Coordinates below are the drawing's own (-1..9, P itself is 0..8); `at`
shifts them by one into array indices, since the drawing's outside ring at
-1 needs a real cell. That shift is also why the placement-tile grid has
origin 1.
"""
import numpy as np
from bruteforce import update_forces
from cores import CoreItem, Pyramid, Base, Upstream, Downstream, DUMMY
from map_of_squares import SquareItem, PlacementStatus


def at(r, c):
    """Drawing coordinates -> array index (the drawing starts at -1)."""
    return (r + 1, c + 1)


def links(*drawing_triples):
    """{(id_in, id_out, id_exit), ...} in drawing coordinates -> array
    indices. DUMMY (chain dies inside the tile) is kept as is."""
    conv = lambda x: x if x == DUMMY else at(*x)
    return {(conv(a), conv(b), conv(c)) for a, b, c in drawing_triples}


PLACEMENT_TILES = {(1, 1)}   # child tile (1,1) of P, drawing cells 3..5 x 3..5
GRID_ORIGIN = 1              # drawing cell 0 is array index 1 (see `at`)


def build_map(shape, chains, placement_tiles=PLACEMENT_TILES):
    """A map of SquareItems whose only conditions are the single links
    along `chains` (each a list of drawing coordinates, forcer -> forced).
    A cell reached by two chains (fan-in) gets two size-1 conditions.
    .forces is rebuilt from .forced_by via update_forces. Sets the global
    placement-tile layout (3x3 tiles, origin GRID_ORIGIN) and marks every
    cell of a placement tile PlacementStatus.candidate, the same way
    test_context_tiles.mark_placement_cores does on a real board."""
    Pyramid.set_placement_tiles(placement_tiles, 3, GRID_ORIGIN)
    m = np.empty(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m[i, j] = SquareItem()
    for ci, cj in placement_tiles:
        for di in range(3):
            for dj in range(3):
                m[GRID_ORIGIN + 3 * ci + di, GRID_ORIGIN + 3 * cj + dj].placement_status = PlacementStatus.candidate
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            a_idx, b_idx = at(*a), at(*b)
            m[b_idx].forced_by.add(frozenset({(a_idx, m[a_idx].placement_status)}))
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

D = DUMMY
EXPECTED_CHILD_LINKS = {
    (0, 0): links(((1, -1), (1, 2), (1, 3)), ((2, -1), (2, 2), (2, 3))),
    # F enters from (2,2) and C's branch from (3,4); both die in here
    (0, 1): links(((1, 2), (1, 5), (1, 6)), ((-1, 5), (0, 5), (0, 6)), ((2, 2), D, D), ((3, 4), D, D)),
    (0, 2): links(((1, 5), (1, 8), (1, 9)), ((0, 5), (0, 8), (-1, 8))),
    (1, 0): set(),
    # every cell of placement tile (1,1) is a candidate: (4,4) starts C,
    # and (4,5) / (3,4) - each its own last cell, linking straight out -
    # start chains of their own
    (1, 1): links(((4, 4), (4, 5), (4, 6)), ((4, 4), (3, 4), (2, 4)),
                  ((4, 5), (4, 5), (4, 6)), ((3, 4), (3, 4), (2, 4))),
    (1, 2): links(((4, 5), (4, 8), (4, 9)), ((6, 7), (5, 8), (5, 9))),
    (2, 0): links(((9, 1), (7, 0), (7, -1))),
    (2, 1): links(((8, 2), (8, 5), (9, 5))),
    (2, 2): links(((9, 7), (6, 7), (5, 7)), ((7, 9), (6, 7), (5, 7))),
}
EXPECTED_CHILD_CANDIDATE_LINKS = {
    tile: {t for t in tl if t[0] in {at(4, 4), at(4, 5), at(3, 4)}}
    for tile, tl in EXPECTED_CHILD_LINKS.items()
}
EXPECTED_PARENT_LINKS = links(
    ((1, -1), (1, 8), (1, 9)),    # A
    ((9, 1), (7, 0), (7, -1)),    # B
    ((4, 4), (4, 8), (4, 9)),     # C, the branch that leaves P
    ((4, 5), (4, 8), (4, 9)),     # C, from candidate (4,5) further along the chain
    ((9, 7), (5, 8), (5, 9)),     # D
    ((7, 9), (5, 8), (5, 9)),     # D
    ((-1, 5), (0, 8), (-1, 8)),   # E
    ((2, -1), D, D),              # F: enters P, dies at (2,3) inside P
    ((3, 4), D, D),               # C: candidate (3,4)'s chain dies at (2,4) inside P
)
EXPECTED_PARENT_CANDIDATE_LINKS = links(((4, 4), (4, 8), (4, 9)), ((4, 5), (4, 8), (4, 9)), ((3, 4), D, D))


def _children(m):
    kids = {}
    for i in range(3):
        for j in range(3):
            k = child_tile(m, i, j)
            Base.update_links(k)
            kids[i, j] = k
    return kids


def test_three_tile_straight_chain_joins_across_child_borders():
    """Smallest case: one straight chain of single links along row 1, from
    (1,2) outside P through three child tiles (cols 3-5, 6-8, 9-11) to
    (1,12) outside P. Each child's id_out is its last cell, which is the
    next child's id_in, so the parent gets one triple, ending at (1,11),
    the last cell inside P, with exit (1,12). No placement tiles. Uses array coordinates directly
    (no `at` shift)."""
    Pyramid.set_placement_tiles(set(), 3, 0)
    m = np.empty((3, 15), dtype=object)
    for i in range(3):
        for j in range(15):
            m[i, j] = SquareItem()
    for j in range(3, 13):
        m[1, j].forced_by = {frozenset({((1, j - 1), PlacementStatus.unknown)})}
    update_forces(m)

    kids = [CoreItem(m[0:3, c:c + 3], margin=None, realm=((0, 3), (c, c + 3))) for c in (3, 6, 9)]
    for k in kids:
        Base.update_links(k)
    assert [k.links for k in kids] == [
        {((1, 2), (1, 5), (1, 6))},
        {((1, 5), (1, 8), (1, 9))},
        {((1, 8), (1, 11), (1, 12))},
    ]

    parent = CoreItem(np.array([kids], dtype=object), margin=None, realm=((0, 3), (3, 12)))
    Upstream.update_links(parent)
    assert parent.links == {((1, 2), (1, 11), (1, 12))}
    assert Pyramid.candidate_links(parent) == set()


def test_child_lists_use_last_cell_inside():
    """Every child tile of the drawing: .links and its candidate part
    exactly as listed in upstream-forced-by.html. Includes the ins that die
    inside a tile, recorded as (id_in, DUMMY, DUMMY) (2026-11-07): F
    entering (0,1) from (2,2), and C's branch entering (0,1) from (3,4).
    Their feeding triples ((2,-1),(2,2),_), ((4,4),(3,4),_) and
    ((3,4),(3,4),_) now join those DUMMY triples; ((8,2),(8,5)) still joins nothing (id_in inside P,
    not a candidate, and no child ends there)."""
    m = build_map((11, 11), CHAINS)
    kids = _children(m)
    assert {t: k.links for t, k in kids.items()} == EXPECTED_CHILD_LINKS
    assert {t: Pyramid.candidate_links(k) for t, k in kids.items()} == EXPECTED_CHILD_CANDIDATE_LINKS


def test_parent_joins_matching_child_pairs_and_drops_the_rest():
    """Parent P over the 3x3 children: matching child triples join into one
    parent triple (A through three tiles, C's surviving branch from both
    candidates (4,4) and (4,5), both D fan-in chains, E entering from
    above); B passes through a single tile. Ins that die inside P stay on
    the list as (id_in, DUMMY, DUMMY) (2026-11-07): F from (2,-1), and candidate
    (3,4) whose only chain dies at (2,4). Candidate (4,4) has a DUMMY-ending
    branch too but also an exit, so it gets no DUMMY. The dangling start
    (G) contributes nothing."""
    m = build_map((11, 11), CHAINS)
    kids = _children(m)
    elements = np.empty((3, 3), dtype=object)
    for (i, j), k in kids.items():
        elements[i, j] = k
    r0, c0 = at(0, 0)
    parent = CoreItem(elements, margin=None, realm=((r0, r0 + 9), (c0, c0 + 9)))
    Upstream.update_links(parent)
    assert parent.links == EXPECTED_PARENT_LINKS
    assert Pyramid.candidate_links(parent) == EXPECTED_PARENT_CANDIDATE_LINKS


def test_build_upstream_reproduces_the_drawing():
    """Upstream.build (the up pass over the whole board, 2026-11-07)
    on the drawing's map: tiles start at array index 1 (the drawing's cell
    0, see `at`), so the 11x11 map holds exactly P's 3x3 child tiles and
    the pyramid is two levels - the 9 children, then P itself. Both levels
    match the expected lists of the tests above."""
    m = build_map((11, 11), CHAINS)
    pyramid = Pyramid(m, 3, origin=GRID_ORIGIN)
    levels = pyramid.upstream.build(pyramid.base.generate_core_items())
    assert pyramid.upstream.pyramid is levels
    assert [lv.shape for lv in levels] == [(3, 3), (1, 1)]
    assert {(i, j): levels[0][i, j].links for i in range(3) for j in range(3)} == EXPECTED_CHILD_LINKS
    top = levels[1][0, 0]
    r0, c0 = at(0, 0)
    assert top.realm == ((r0, r0 + 9), (c0, c0 + 9))
    assert top.links == EXPECTED_PARENT_LINKS


def test_onepass_writes_the_updated_base_level_back_to_base():
    """Pyramid.onepass (2026-11-07): Base generates the base-level
    CoreItems, Upstream builds .pyramid up to the top, Downstream runs from
    the top down and writes the updated base level back to
    Base.map_onepass - the same array as Upstream.pyramid[0], updated in
    place.

    On the drawing (P is the top, so it receives nothing from above), every
    base tile keeps its up-pass triples and gains only what siblings tell
    each other about candidate (4,4):
    - (1,2): (4,4)'s chain, which runs through (1,2);
    - (0,1): (4,4)'s other branch, which enters (0,1) via (3,4) and dies.
    Nothing else - in particular (1,1) no longer receives copies of P's own
    triples (the push_down of before 2026-11-07, see push_down)."""
    m = build_map((11, 11), CHAINS)
    pyramid = Pyramid(m, 3, origin=GRID_ORIGIN)
    onepass = pyramid.onepass()

    assert onepass is pyramid.base.map_onepass
    assert onepass is pyramid.upstream.pyramid[0]
    assert pyramid.downstream.pyramid is pyramid.upstream.pyramid

    gained = {
        (0, 1): links(((4, 4), D, D)),
        (1, 2): links(((4, 4), (4, 8), (4, 9))),
    }
    for i in range(3):
        for j in range(3):
            assert onepass[i, j].links == EXPECTED_CHILD_LINKS[i, j] | gained.get((i, j), set()), (i, j)


Z_DRAWN = (-1, -1)  # Z's cell in drawing coordinates (outside P)
Z = at(*Z_DRAWN)    # a candidate outside P, known to P only through P's parent
Z_INHERITED = {(Z, at(1, -1)), (Z, at(-1, 5)), (Z, at(2, -1))}
EXPECTED_Z_GAINED = {
    # A and F leave (0,0) at its last cells (1,2) and (2,2)
    (0, 0): links((Z_DRAWN, (1, 2), (1, 3)), (Z_DRAWN, (2, 2), (2, 3))),
    # A and E pass through (0,1); F enters at (2,3) and dies, but Z has exits
    # here, so no DUMMY ("partly dying starts get no dummy"); plus the
    # sibling fact about (4,4) from the test above
    (0, 1): links((Z_DRAWN, (1, 5), (1, 6)), (Z_DRAWN, (0, 5), (0, 6)), ((4, 4), D, D)),
    # A and E leave P through (0,2)
    (0, 2): links((Z_DRAWN, (1, 8), (1, 9)), (Z_DRAWN, (0, 8), (-1, 8))),
    (1, 2): links(((4, 4), (4, 8), (4, 9))),
}


def test_push_down_copies_an_outside_candidate_into_every_child_its_chain_touches():
    """The Z case (Ivo, 2026-11-07: "all candidates must be copied down,
    including those outside P, like Z"): P's own parent told P that
    candidate Z, outside P, reaches the cells where chains A, E and F enter
    P - (Z, (1,-1)), (Z, (-1,5)), (Z, (2,-1)). P's down step
    (update_aliases_down with these as `inherited`; push_down walks them)
    delivers Z to every child those chains enter or pass through, and each
    child derives its own triples for Z: its last cell and exit, as for any
    start. downstream-z.html (project root) shows the same."""
    m = build_map((11, 11), CHAINS)
    pyramid = Pyramid(m, 3, origin=GRID_ORIGIN)
    levels = pyramid.upstream.build(pyramid.base.generate_core_items())
    P = levels[1][0, 0]

    inbox = Downstream.update_aliases_down(P, 1, Z_INHERITED)

    z_told = {idx: {cell for origin, cell in inbox[idx] if origin == Z} for idx in np.ndindex(3, 3)}
    assert z_told == {
        (0, 0): {at(1, -1), at(1, 2), at(2, -1), at(2, 2)},
        (0, 1): {at(-1, 5), at(0, 5), at(1, 2), at(1, 5), at(2, 2)},
        (0, 2): {at(0, 5), at(0, 8), at(1, 5), at(1, 8)},
        (1, 0): set(), (1, 1): set(), (1, 2): set(),
        (2, 0): set(), (2, 1): set(), (2, 2): set(),
    }
    for i in range(3):
        for j in range(3):
            assert levels[0][i, j].links == EXPECTED_CHILD_LINKS[i, j] | EXPECTED_Z_GAINED.get((i, j), set()), (i, j)
