"""Exploratory tests for context_tiles.py - concrete small situations of
simultaneous placement (one placement per active core of a supergrid sweep,
see context_tiles.py's own docstring for the four things known at each real
step), worked out by hand/by inspection first, so the algorithm in
context_tiles.py gets built from what these show is actually safe to assume,
rather than designed up front and tested after the fact. Expect this file to
grow one concrete case at a time.
"""
import random

import numpy as np

from conditions import initial_map, forced_by_on_realm
from closure import place_square
from bruteforce import (compute_conditions, retag_forced_by, prune_same_core_candidates,
                         unblock_path, collapse_synced_ids)
from cores import CoreItem, DeviceMemory, Pyramid, Base, Upstream, Downstream, DUMMY
from map_of_squares import StateEnum, PlacementStatus, SquareItem


def build_framed_map(n, M):
    """An n x n grid of M x M cores, with a chosen frame forming the
    outermost ring of the WHOLE board (row/col 0 and row/col N-1) - N = M*n
    exactly, no margin or frame-only cells beyond the core grid at all; the
    frame overlaps the core grid's own outermost ring rather than sitting
    outside it (confirmed deliberate - core (0,0) spans board rows/cols
    0..M-1 directly, so its own row 0 and column 0 ARE the frame's row 0 and
    column 0; its free interior shrinks to rows/cols 1..M-1,
    (M-1)x(M-1) cells, once the frame places - same for every other
    edge/corner core).

    The four sides' own frame positions aren't all independently placeable:
    near each corner, one side's chosen cell diagonally blocks the adjacent
    side's own frame position before it's placed there (now at the true
    corner itself, e.g. (0,1) blocks (1,0), one cell further out than the
    (1,2)/(2,1) collision this had before the frame moved to N=M*n) -
    confirmed, not assumed. Handled the same way as before: skip a frame
    position if it's no longer free by the time this reaches it - that
    position ends up part of the blocked ring on its own, without ever
    needing to be chosen itself.

    That same corner-collision handling can leave a cell with a
    fully-satisfied condition (a "seat") before any further placement at
    all. Rather than leave that as an artifact of the frame construction,
    it's filled the same way any other forced cell would be: actually
    chosen, via place_square, same as everything else here. Iterated to a
    fixed point (filling one seat's diagonal-blocking side effect could in
    principle complete another).

    Returns (m, N, skipped, filled_seats) - skipped is the list of frame
    positions that were already non-free by the time they were reached
    (includes each corner's own duplicate row/column reference, which is
    never placed twice); filled_seats is the list of positions chosen this
    way.
    """
    N = M * n
    m = initial_map((N, N))
    frame_positions = list(dict.fromkeys(
        [(0, c) for c in range(N)] + [(N - 1, c) for c in range(N)]
        + [(r, 0) for r in range(N)] + [(r, N - 1) for r in range(N)]
    ))
    skipped = []
    for pos in frame_positions:
        if m[pos].state == StateEnum.free:
            place_square(m, pos)
        else:
            skipped.append(pos)

    filled_seats = []
    while True:
        seats = [(i, j) for i in range(N) for j in range(N)
                  if m[i, j].state == StateEnum.free
                  and any(all(m[p].state == StateEnum.chosen for p, _status in cond)
                          for cond in m[i, j].forced_by)]
        if not seats:
            break
        for pos in seats:
            if m[pos].state == StateEnum.free:
                place_square(m, pos)
                filled_seats.append(pos)

    return m, N, skipped, filled_seats


def _report_frame(m, N, n, M, skipped, filled_seats):
    core_free = sum(1 for i in range(N) for j in range(N)
                     if m[i, j].state == StateEnum.free)
    core_blocked = sum(1 for i in range(N) for j in range(N)
                        if m[i, j].state == StateEnum.blocked)
    print(f'n={n} cores of {M}x{M}: N={N}, board area={N ** 2}, '
          f'free={core_free}, blocked={core_blocked}, '
          f'frame positions skipped (already blocked)={len(skipped)}: {skipped}, '
          f'corner seats filled={len(filled_seats)}: {filled_seats}')


def test_framed_map_3x3_cores():
    """n=7 M=3 cores, N=M*n=21 exactly - the chosen frame IS the board's own
    outermost ring (row/col 0 and N-1), overlapping the outermost ring of
    cores rather than sitting outside them. Every frame position ends up
    either chosen (placed) or blocked (skipped - a neighbouring frame
    placement, now right at the corner itself, already blocked it)."""
    n, M = 7, 3
    m, N, skipped, filled_seats = build_framed_map(n, M)
    assert N == 21
    for c in range(N):
        assert m[0, c].state in (StateEnum.chosen, StateEnum.blocked)
        assert m[N - 1, c].state in (StateEnum.chosen, StateEnum.blocked)
    for r in range(N):
        assert m[r, 0].state in (StateEnum.chosen, StateEnum.blocked)
        assert m[r, N - 1].state in (StateEnum.chosen, StateEnum.blocked)
    _report_frame(m, N, n, M, skipped, filled_seats)


def placement_core_indices(n):
    """Core-indices (ci, cj) of the fixed "placement" cores within an n x n
    core grid: (1, 1), (1+4I, 1+4J) for I, J = 0, 1, ... Every other core is
    a "transit" core - we are no longer free to choose a cell anywhere on
    the board; only a placement core's own interior is ever a candidate to
    have one of its cells actually chosen, a transit core exists only to
    connect ins to outs. For n=7 this is {1, 5}x{1, 5} - 4 cores, not the
    {1,3,5}x{1,3,5}/9-core (1+2I,1+2J) spacing used earlier (corrected
    2026-10-22 - 1+4I, not 1+2I)."""
    return {(ci, cj) for ci in range(1, n, 4) for cj in range(1, n, 4)}


def test_transit_core_has_at_most_one_placement_neighbor():
    """The whole point of the 1+4I/1+4J spacing (not the earlier, wrong 1+2I/
    1+2J one, which packed placement cores too close together): every transit
    core is 8-connected-adjacent (Chebyshev distance 1 in core-index space) to
    at most one placement core. This is what will let a transit core's own
    "out" unambiguously mean "towards the one placement core it borders" once
    ins/outs get built on top of this split - checked generically here, not
    just eyeballed for n=7."""
    n = 7
    placement = placement_core_indices(n)

    def neighbors(ci, cj):
        return {(ci + di, cj + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0) and 0 <= ci + di < n and 0 <= cj + dj < n}

    for ci in range(n):
        for cj in range(n):
            if (ci, cj) in placement:
                continue
            adjacent_placement = [p for p in neighbors(ci, cj) if p in placement]
            assert len(adjacent_placement) <= 1, \
                f"transit core ({ci},{cj}) borders {len(adjacent_placement)} placement cores: {adjacent_placement}"


def mark_placement_cores(m, n, M, s=0):
    """Set .placement_status on every cell of the board: PlacementStatus.
    candidate for every cell inside one of the fixed placement cores (see
    placement_core_indices), PlacementStatus.unknown for every other cell -
    every transit core's own cells, including the frame (which is not part
    of any placement core at this n/M). Independent of .state; meant to be
    run once on the "original state" (right after build_framed_map, before
    any further placement) to set up the renamed .placement_status field
    (SquareItem.eligible before 2026-10-22) for the fixed placement/transit
    core split. Returns the set of placement-core indices used, for the
    caller to assert against.

    Also sets the global layout in cores (Pyramid.set_placement_tiles, 2026-11-07),
    so Pyramid.is_candidate answers the same for any id, on or off this map.
    """
    rows, cols = m.shape
    placement_cores = placement_core_indices(n)
    Pyramid.set_placement_tiles(placement_cores, M, s)
    for i in range(rows):
        for j in range(cols):
            m[i, j].placement_status = PlacementStatus.unknown
    for ci, cj in placement_cores:
        r0, c0 = s + ci * M, s + cj * M
        for di in range(M):
            for dj in range(M):
                m[r0 + di, c0 + dj].placement_status = PlacementStatus.candidate
    return placement_cores


def place_chessboard_cores(m, n, M, s, rng, origin_bound=None, white=False):
    """One random cell in every core of the n x n core grid (each core's own
    board-relative origin is (s + ci*M, s + cj*M)) whose own core-indices
    ci, cj fall on the "black" chessboard colour - the one (1, 1) and (0, 0)
    share, ci == cj (mod 2) - or, if white=True, the opposite colour,
    ci != cj (mod 2) (e.g. (0,1), (1,0), (1,2)...). Black is exactly "the
    mod(j,2) rule from before": i = 2*I + (J mod 2), j = J, confirmed earlier
    to enumerate exactly the 25 same-parity cores of a 7x7 core grid -
    restated here directly as the ci==cj(mod 2) condition it's equivalent
    to, rather than re-deriving the I/J formula. White is everything black
    leaves out - the remaining 24 cores of a 7x7 grid.

    origin_bound, if given, additionally restricts to cores whose own origin
    (in both row and column) is <= origin_bound - a board-coordinate bound,
    not a core-index one. With the core grid's own origin now s=0 (see
    build_framed_map - core origins are plain ci*M/cj*M), origin_bound=M
    picks out exactly core-indices {0, 1} for any M: origins are 0, M, 2M,
    ... and M itself sits exactly on the boundary between the first two and
    the third.

    Places via place_square, not a raw .state assignment, so each placement
    actually diagonally-blocks its own free neighbours too, same as a real
    placement would - a raw assignment was the earlier behaviour throughout
    this whole session's bare-board experiments, confirmed as a real gap
    once place_square was in use elsewhere in this same file (the frame,
    build_framed_map's own corner-seat fill), not a deliberate choice.

    A core's own M*M cells aren't always all free by the time this reaches
    it - e.g. a core right against the frame can have most of its cells
    already blocked by the frame's own diagonal reach (confirmed: core (0,0)
    on the n=7/M=3 map has only 2 of 9 cells still free) - so this picks
    uniformly among whichever of a core's own cells are still free, not a
    raw random offset into the full M*M block; a core left with no free
    cells at all is skipped and reported separately, not an error.
    """
    placed = []
    cores_with_no_free_cell = []
    for ci in range(n):
        for cj in range(n):
            if (ci % 2 == cj % 2) == white:
                continue
            r0, c0 = s + ci * M, s + cj * M
            if origin_bound is not None and (r0 > origin_bound or c0 > origin_bound):
                continue
            free_cells = [(r0 + di, c0 + dj) for di in range(M) for dj in range(M)
                          if m[r0 + di, c0 + dj].state == StateEnum.free]
            if not free_cells:
                cores_with_no_free_cell.append((ci, cj))
                continue
            cell = rng.choice(free_cells)
            place_square(m, cell)
            placed.append(cell)
    return placed, cores_with_no_free_cell


def count_links_and_seats(m, N):
    """Read-only scan, no mutation, no compute_conditions call - a 'link' is
    a .forced_by condition with exactly 2 of 3 members chosen, a 'seat' one
    with all 3 - thresholds confirmed empirically earlier against Ivo's own
    "no links" claim on a sparse initial state (>=1 gave false positives;
    ==2 gave exactly 0, matching)."""
    links = seats = 0
    for i in range(N):
        for j in range(N):
            if m[i, j].state != StateEnum.free:
                continue
            for cond in m[i, j].forced_by:
                n_chosen = sum(1 for pos, _status in cond if m[pos].state == StateEnum.chosen)
                if n_chosen == 2:
                    links += 1
                elif n_chosen == 3:
                    seats += 1
    return links, seats


def test_chessboard_placement_3x3_cores():
    """The chessboard-colour initial random placement (same rule as the
    earlier bare-board experiments) applied inside the framed 3x3-core map."""
    n, M = 7, 3
    m, N, _, _ = build_framed_map(n, M)
    rng = random.Random(0)
    placed, no_free = place_chessboard_cores(m, n, M, s=0, rng=rng)
    links, seats = count_links_and_seats(m, N)
    print(f'n={n} cores of {M}x{M}: {len(placed)} cells placed on chessboard '
          f'cores, links={links}, seats={seats}, cores with no free cell left='
          f'{no_free}')
    assert len(placed) + len(no_free) == 25


def test_chessboard_placement_6x6_cores():
    """Same chessboard placement, inside the framed 6x6-core map."""
    n, M = 7, 6
    m, N, _, _ = build_framed_map(n, M)
    rng = random.Random(0)
    placed, no_free = place_chessboard_cores(m, n, M, s=0, rng=rng)
    links, seats = count_links_and_seats(m, N)
    print(f'n={n} cores of {M}x{M}: {len(placed)} cells placed on chessboard '
          f'cores, links={links}, seats={seats}, cores with no free cell left='
          f'{no_free}')
    assert len(placed) + len(no_free) == 25


def test_chessboard_placement_white_3x3_cores():
    """Same as test_chessboard_placement_3x3_cores, but the white cores (the
    24 cores black leaves out) instead of black."""
    n, M = 7, 3
    m, N, _, _ = build_framed_map(n, M)
    rng = random.Random(0)
    placed, no_free = place_chessboard_cores(m, n, M, s=0, rng=rng, white=True)
    links, seats = count_links_and_seats(m, N)
    print(f'n={n} cores of {M}x{M}, white: {len(placed)} cells placed on '
          f'chessboard cores, links={links}, seats={seats}, cores with no '
          f'free cell left={no_free}')
    assert len(placed) + len(no_free) == 24


def test_chessboard_placement_white_6x6_cores():
    """Same as test_chessboard_placement_6x6_cores, but white cores."""
    n, M = 7, 6
    m, N, _, _ = build_framed_map(n, M)
    rng = random.Random(0)
    placed, no_free = place_chessboard_cores(m, n, M, s=0, rng=rng, white=True)
    links, seats = count_links_and_seats(m, N)
    print(f'n={n} cores of {M}x{M}, white: {len(placed)} cells placed on '
          f'chessboard cores, links={links}, seats={seats}, cores with no '
          f'free cell left={no_free}')
    assert len(placed) + len(no_free) == 24


def test_chessboard_placement_restricted_3x3_cores():
    """Chessboard placement restricted via origin_bound=M - picks out
    core-indices {0, 1} only. Does NOT give 0 seats, unlike the
    earlier (incorrect) version of this test: that version placed via a raw
    .state assignment, without place_square's diagonal-blocking side effect,
    so it never actually interacted with the frame. With real blocking
    (fixed once place_chessboard_cores started using place_square), core
    (0,0)'s own placement can land close enough to the frame's own top-left
    corner to complete a seat there - confirmed concretely: cell (4,3),
    placed in core (0,0), combines with the frame's own chosen cells
    (1,3)..(1,6) and the filled corner seat (3,3) to force (3,5). A genuine
    instance of "it does not happen in a vacuum" - the frame is real prior
    structure a nearby placement can close a gap in, not a bug."""
    n, M = 7, 3
    m, N, _, _ = build_framed_map(n, M)
    rng = random.Random(0)
    placed, no_free = place_chessboard_cores(m, n, M, s=0, rng=rng, origin_bound=M)
    links, seats = count_links_and_seats(m, N)
    print(f'n={n} cores of {M}x{M}, restricted: {len(placed)} cells placed '
          f'({placed}), links={links}, seats={seats}, no free cell in={no_free}')


def test_chessboard_placement_restricted_6x6_cores():
    """Same restriction, scaled for M=6: origin_bound=M picks out the same
    relative core-indices {0, 1} as the 3x3-core version above."""
    n, M = 7, 6
    m, N, _, _ = build_framed_map(n, M)
    rng = random.Random(0)
    placed, no_free = place_chessboard_cores(m, n, M, s=0, rng=rng, origin_bound=M)
    links, seats = count_links_and_seats(m, N)
    print(f'n={n} cores of {M}x{M}, restricted: {len(placed)} cells placed '
          f'({placed}), links={links}, seats={seats}, no free cell in={no_free}')
    assert seats == 0


def test_placement_cores_3x3():
    """Original state (frame built, no core placement yet) of the 3x3-core
    map, with .placement_status set per the fixed placement/transit core
    split - PlacementStatus.candidate inside the 4 placement cores
    ({1,5}x{1,5} out of the 7x7 core grid), PlacementStatus.unknown
    everywhere else, including the frame."""
    n, M = 7, 3
    m, N, _, _ = build_framed_map(n, M)
    placement_cores = mark_placement_cores(m, n, M, s=0)
    assert placement_cores == {(ci, cj) for ci in (1, 5) for cj in (1, 5)}
    candidate = sum(1 for i in range(N) for j in range(N)
                     if m[i, j].placement_status == PlacementStatus.candidate)
    unknown = sum(1 for i in range(N) for j in range(N)
                   if m[i, j].placement_status == PlacementStatus.unknown)
    assert candidate == len(placement_cores) * M * M
    assert candidate + unknown == N * N


def test_placement_cores_6x6():
    """Same placement/transit core split, scaled for the 6x6-core map - same
    4 placement-core indices ({1,5}x{1,5} out of the same 7x7 core grid),
    just M=6 cells per core instead of 3."""
    n, M = 7, 6
    m, N, _, _ = build_framed_map(n, M)
    placement_cores = mark_placement_cores(m, n, M, s=0)
    assert placement_cores == {(ci, cj) for ci in (1, 5) for cj in (1, 5)}
    candidate = sum(1 for i in range(N) for j in range(N)
                     if m[i, j].placement_status == PlacementStatus.candidate)
    unknown = sum(1 for i in range(N) for j in range(N)
                   if m[i, j].placement_status == PlacementStatus.unknown)
    assert candidate == len(placement_cores) * M * M
    assert candidate + unknown == N * N


def test_is_candidate_matches_placement_status_everywhere():
    """Pyramid.is_candidate (global placement-tile layout, 2026-11-07) gives
    exactly the answer mark_placement_cores writes into .placement_status,
    for every cell of the board - for both tile sizes and with a non-zero
    grid origin s - and answers for ids off the board too, without a map:
    those are in no placement tile here, since the placement tiles
    (1+4I, 1+4J) never touch the board's edge."""
    for M, s in ((3, 0), (6, 0), (3, 2)):
        n = 7
        m, N, _, _ = build_framed_map(n, M)
        mark_placement_cores(m, n, M, s=s)
        mismatches = [(i, j) for i in range(N) for j in range(N)
                      if Pyramid.is_candidate((i, j)) != (m[i, j].placement_status == PlacementStatus.candidate)]
        assert mismatches == [], (M, s, mismatches[:5])
        assert not Pyramid.is_candidate((-1, -1))
        assert not Pyramid.is_candidate((N + 5, 0))

    # explicit layout, independent of any map: tile (1,1) of 3x3 tiles at origin 0
    Pyramid.set_placement_tiles({(1, 1)}, 3, 0)
    assert [Pyramid.is_candidate((r, c)) for r, c in ((3, 3), (5, 5), (2, 3), (3, 6), (4, 4))] == \
           [True, True, False, False, True]


def test_framed_map_6x6_cores():
    """n=7 M=6 cores, N=M*n=42 exactly - same discipline as
    test_framed_map_3x3_cores."""
    n, M = 7, 6
    m, N, skipped, filled_seats = build_framed_map(n, M)
    assert N == 42
    for c in range(N):
        assert m[0, c].state in (StateEnum.chosen, StateEnum.blocked)
        assert m[N - 1, c].state in (StateEnum.chosen, StateEnum.blocked)
    for r in range(N):
        assert m[r, 0].state in (StateEnum.chosen, StateEnum.blocked)
        assert m[r, N - 1].state in (StateEnum.chosen, StateEnum.blocked)
    _report_frame(m, N, n, M, skipped, filled_seats)


def _blocked_member_violations(m, N):
    """Every (cell, blocked_id) pair where cell's own .forced_by still
    names blocked_id as a condition member - should be empty after
    compute_conditions, which is specifically responsible for dropping a
    condition the moment any of its own members is blocked (that placement
    can never happen, so the whole condition is false - see its own
    docstring)."""
    return [((i, j), pos) for i in range(N) for j in range(N)
            for cond in m[i, j].forced_by for pos, _status in cond
            if m[pos].state == StateEnum.blocked]


def test_no_blocked_members_survive_compute_conditions_3x3():
    """Real placement (frame + white chessboard cores) does create blocked
    ids inside some cells' .forced_by - place_square's own diagonal blocking
    only touches .state, nothing prunes .forced_by in response until
    compute_conditions runs. Confirms both halves: a real violation exists
    beforehand, and compute_conditions removes every one of them, not just
    some."""
    n, M = 7, 3
    m, N, _, _ = build_framed_map(n, M)
    mark_placement_cores(m, n, M, s=0)
    rng = random.Random(0)
    place_chessboard_cores(m, n, M, s=0, rng=rng, white=True)
    assert _blocked_member_violations(m, N), \
        "expected at least one blocked id inside a forced_by tuple before compute_conditions"
    compute_conditions(m)
    assert _blocked_member_violations(m, N) == []


def test_no_blocked_members_survive_compute_conditions_6x6():
    """Same check, scaled for the 6x6-core map with 3x3 placement cores."""
    n, M, P = 7, 6, 3
    m, N, _, _ = build_framed_map(n, M)
    mark_placement_cores(m, n, M, s=0)
    n_p = N // P
    rng = random.Random(0)
    place_chessboard_cores(m, n_p, P, s=0, rng=rng, white=True)
    assert _blocked_member_violations(m, N), \
        "expected at least one blocked id inside a forced_by tuple before compute_conditions"
    compute_conditions(m)
    assert _blocked_member_violations(m, N) == []


def build_swept_map(n, M, s=0, seed=0):
    """The scenario used throughout this file's own CoreItem.forced_by/
    .placement_reach/.links exploration (2026-10-23..25): the framed map,
    placement cores marked and tagged, then placed via the original 5-phase
    sweep from way back in this same file's own history (odd-odd initial 9
    cores, then the 4 even-even quadrant shift sub-phases i=4I+shift/
    j=4J+shift), compute_conditions run after every phase (so .forced_by
    actually reduces triple->pair->single as cells get chosen - a real bug
    in this file's own early verification scripts was forgetting this step
    and getting all-zero results everywhere), then
    prune_same_core_candidates once at the end.

    Consolidates what was, before this function existed, retyped by hand in
    essentially every ad hoc verification script across several turns of
    this session - see squareplacement_superlattice_refactor memory note,
    2026-10-25 entries, for the exploratory history this formalizes.

    Returns (m, N).
    """
    m, N, _, _ = build_framed_map(n, M)
    mark_placement_cores(m, n, M, s=s)
    retag_forced_by(m)
    rng = random.Random(seed)
    # range(n) is generous for every I/J here (always enough to cover an
    # n x n core grid regardless of n) - the out-of-range candidates it
    # produces are discarded by the 0<=ci<n filter below anyway, same as
    # the original range(3)/(0,1) bounds (sized for exactly n=7) already
    # relied on that filter to do the real trimming. Widened 2026-10-25 so
    # this generalizes to other n instead of silently under-covering a
    # bigger board (caught when n=9 produced almost no activity at all).
    phases = [
        [(2 * I + 1, 2 * J + 1) for I in range(n) for J in range(n)],
        [(4 * I, 4 * J) for I in range(n) for J in range(n)],
        [(2 + 4 * I, 2 + 4 * J) for I in range(n) for J in range(n)],
        [(2 + 4 * I, 4 * J) for I in range(n) for J in range(n)],
        [(4 * I, 2 + 4 * J) for I in range(n) for J in range(n)],
    ]
    for idx in phases:
        idx = [(ci, cj) for ci, cj in idx if 0 <= ci < n and 0 <= cj < n]
        for ci, cj in idx:
            r0, c0 = s + ci * M, s + cj * M
            free_cells = [(r0 + di, c0 + dj) for di in range(M) for dj in range(M)
                          if m[r0 + di, c0 + dj].state == StateEnum.free]
            if free_cells:
                place_square(m, rng.choice(free_cells))
        compute_conditions(m)
    prune_same_core_candidates(m, M, s=s)
    return m, N


def _base_tile(m, ci, cj, M, s=0):
    """One base-level CoreItem for core-index (ci, cj), M x M cells,
    origin s - same (s + ci*M, s + cj*M) convention as build_framed_map/
    mark_placement_cores/place_chessboard_cores throughout this file."""
    r0, c0 = s + ci * M, s + cj * M
    return CoreItem(m[r0:r0 + M, c0:c0 + M], margin=None, realm=((r0, r0 + M), (c0, c0 + M)))


def test_links_id_out_is_last_cell_inside_own_tile():
    """"Last cell inside" convention (Ivo, 2026-11-07 - replaces the
    2026-10-25 rule that id_out is always OUTSIDE the tile), for every one
    of the 49 base tiles and every (id_in, id_out, id_exit) triple in
    .links (named .forced_by, holding pairs, until 2026-11-07):
    - id_out is INSIDE the tile, id_exit is OUTSIDE it, and id_exit is
      really forced by id_out alone (a size-1 condition {id_out} on
      id_exit's own cell);
    - or the chain died inside the tile: (id_in, DUMMY, DUMMY);
    - id_in is either outside the tile or a candidate inside it - never an
      ordinary cell of the tile itself."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    def core_of(id_):
        row, col = id_
        return row // M, col // M

    def single_forcers(cell):
        return {next(iter(cond))[0] for cond in m[cell].forced_by if len(cond) == 1}

    violations = []
    for ci in range(n):
        for cj in range(n):
            core = _base_tile(m, ci, cj, M)
            Base.update_links(core)
            for id_in, id_out, id_exit in core.links:
                if core_of(id_in) == (ci, cj) and m[id_in].placement_status != PlacementStatus.candidate:
                    violations.append(("id_in inside but not a candidate", (ci, cj), id_in))
                if id_out == DUMMY or id_exit == DUMMY:
                    if (id_out, id_exit) != (DUMMY, DUMMY):
                        violations.append(("half DUMMY", (ci, cj), (id_in, id_out, id_exit)))
                    continue
                if core_of(id_out) != (ci, cj):
                    violations.append(("id_out outside own tile", (ci, cj), id_out))
                if core_of(id_exit) == (ci, cj):
                    violations.append(("id_exit inside own tile", (ci, cj), id_exit))
                if id_out not in single_forcers(id_exit):
                    violations.append(("no single link id_out -> id_exit", (ci, cj), (id_out, id_exit)))
    assert violations == []


def test_candidate_links_start_at_candidates_of_the_map():
    """Pyramid.candidate_links() (replaces the .placement_reach field,
    removed 2026-11-07) is exactly the part of .links whose id_in the map
    itself marks PlacementStatus.candidate - is_candidate's global layout
    and the map's own .placement_status agree, for every one of the 49 base
    tiles - and it is not empty everywhere."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    checked_nonempty = False
    for ci in range(n):
        for cj in range(n):
            core = _base_tile(m, ci, cj, M)
            Base.update_links(core)
            expected = {t for t in core.links
                        if m[t[0]].placement_status == PlacementStatus.candidate}
            assert Pyramid.candidate_links(core) == expected
            if expected:
                checked_nonempty = True
    assert checked_nonempty, "expected at least one tile with a candidate-started triple"


def test_hierarchy_composition_matches_monolithic_computation():
    """The core regression test for 'follow the hierarchy up': a CoreItem
    built by composing several levels of smaller CoreItems must give
    EXACTLY the same .forced_by as a single CoreItem built directly
    (base-level style) over that same bigger realm in one pass, at several
    different realms/offsets.

    This is the exact check that caught a real bug in _compose_up's first
    version (2026-10-25): it treated every id that was ever an origin for
    ANY child as unconditionally still a valid origin at every higher
    level, which is correct for a candidate but wrong for a plain external-entry origin in .forced_by, since a
    non-candidate id that was merely external to one small child can become
    interior to a bigger composed realm - see squareplacement_superlattice_
    refactor memory note, 2026-10-25 entry, for the worked example and the
    require_external_origin fix in _compose_up/update_links.
    """
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    base = np.empty((n, n), dtype=object)
    for ci in range(n):
        for cj in range(n):
            core = _base_tile(m, ci, cj, M)
            Base.update_links(core)
            base[ci, cj] = core

    def composed_vs_monolithic(sub, realm):
        composed = CoreItem(sub, margin=None, realm=realm)
        Upstream.update_links(composed)
        (r0, r1), (c0, c1) = realm
        monolithic = CoreItem(m[r0:r1, c0:c1], margin=None, realm=realm)
        Base.update_links(monolithic)
        return composed.links == monolithic.links

    assert composed_vs_monolithic(base[0:6, 0:6], ((0, 18), (0, 18)))
    assert composed_vs_monolithic(base[1:4, 2:5], ((1 * M, 4 * M), (2 * M, 5 * M)))
    assert composed_vs_monolithic(base[2:5, 1:4], ((2 * M, 5 * M), (1 * M, 4 * M)))
    assert composed_vs_monolithic(base[0:7, 0:7], ((0, N), (0, N)))


def test_build_upstream_every_level_matches_direct_computation():
    """Upstream.build on the 5-phase-swept 7x7-tile map: 7x7 base
    tiles, then 3x3 parents (the last row/column of groups holds only one
    tile, 7 = 3+3+1), then the 1x1 top over the whole 21x21 board. Every
    CoreItem at every level has exactly the .links a single base-level
    CoreItem computed directly over the same realm would have - the
    composition is lossless all the way up, including the smaller groups
    at the edge."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    levels = Pyramid.upward(DeviceMemory(m, M))
    assert [lv.shape for lv in levels] == [(7, 7), (3, 3), (1, 1)]
    assert levels[1][2, 2].elements.shape == (1, 1)
    assert levels[-1][0, 0].realm == ((0, N), (0, N))
    for lv in levels:
        for core in lv.flat:
            (r0, r1), (c0, c1) = core.realm
            direct = CoreItem(m[r0:r1, c0:c1], margin=None, realm=core.realm)
            Base.update_links(direct)
            assert core.links == direct.links, core.realm


def _state_only_copy(m):
    """What the CPU main hands to the device (2026-11-08): a fresh map of
    SquareItems with nothing set but .state, copied from m."""
    fresh = np.empty(m.shape, dtype=object)
    for idx in np.ndindex(m.shape):
        fresh[idx] = SquareItem()
        fresh[idx].state = m[idx].state
    return fresh


def test_init_map_from_state_matches_cpu_preprocessing():
    """Base.init_map (2026-11-08, "move all map preprocessing to Base"):
    from a map with only .state set, plus the placement-tile layout, it
    derives exactly the .forced_by, .forces and .placement_status the
    CPU-side preprocessing of build_swept_map produced (initial_map,
    mark_placement_cores, retag_forced_by, compute_conditions after every
    placement phase, prune_same_core_candidates) - on every cell. One
    reduction with the final states is enough, since states only ever go
    free -> chosen / blocked."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)          # also sets the placement-tile layout
    mem = DeviceMemory(_state_only_copy(m), M)
    Base.init_map(mem)
    for idx in np.ndindex(m.shape):
        cell, ref = mem.map[idx], m[idx]
        assert cell.state == ref.state, idx
        assert cell.forced_by == ref.forced_by, idx
        assert cell.forces == ref.forces, idx
        assert cell.placement_status == ref.placement_status, idx


def test_onepass_from_state_only_matches_preprocessed_map():
    """The CPU main end to end (Pyramid.onepass, 2026-11-08): a map with only
    .state set goes into DeviceMemory, Base.init_map derives the conditions,
    then the up and down passes run. Its .map_after has, tile by tile,
    exactly the .links of the same passes run on the CPU-preprocessed map."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    ref = DeviceMemory(m, M)
    Base.suspend_contradictory(ref)     # onepass's step after init_map (2026-11-12)
    Pyramid.upward(ref)
    Pyramid.downward(ref)

    mem = DeviceMemory(_state_only_copy(m), M)
    after = Pyramid.onepass(mem)
    assert after is mem.map_after and after.shape == (n, n)
    assert [lv.shape for lv in mem.pyramid] == [lv.shape for lv in ref.pyramid]
    for idx in np.ndindex(after.shape):
        assert after[idx].links == ref.map_after[idx].links, idx


def test_nary_conditions_cannot_resolve_in_a_single_placement_round():
    """Ivo's hypothesis (2026-10-23): "n-aries with n>1 cannot propagate on
    their own... all that n-aries do is to link paths together." Checked by
    snapshotting every size>=2 .forced_by condition's currently-chosen
    members before one round of placement (one cell per
    PlacementStatus.candidate core, all simultaneous - the real
    placement-core discipline, not the denser historical chessboard scheme,
    which does NOT respect it) and comparing after: no condition should
    ever gain more than 1 newly-chosen member in a single such round.

    Confirmed across 3 successive rounds, not just the first - a pair/
    triple can only ever shed one member per round, so full resolution
    necessarily spans multiple rounds. See squareplacement_superlattice_
    refactor memory note, 2026-10-23 entries, for the full history
    (including a first, wrong measurement using the chessboard scheme that
    was later corrected)."""
    n, M = 7, 3
    m, N, _, _ = build_framed_map(n, M)
    mark_placement_cores(m, n, M, s=0)
    retag_forced_by(m)
    rng = random.Random(0)

    def chosen_members(condition):
        return {pos for pos, _status in condition if m[pos].state == StateEnum.chosen}

    def snapshot():
        snap = {}
        for i in range(N):
            for j in range(N):
                for cond in m[i, j].forced_by:
                    if len(cond) >= 2 and cond not in snap:
                        snap[cond] = chosen_members(cond)
        return snap

    placement_cores = placement_core_indices(n)

    for _round in range(3):
        before = snapshot()
        for ci, cj in placement_cores:
            r0, c0 = ci * M, cj * M
            free_cells = [(r0 + di, c0 + dj) for di in range(M) for dj in range(M)
                          if m[r0 + di, c0 + dj].state == StateEnum.free]
            if free_cells:
                place_square(m, rng.choice(free_cells))
        max_jump = 0
        for i in range(N):
            for j in range(N):
                if m[i, j].state != StateEnum.free:
                    continue
                for cond in m[i, j].forced_by:
                    if len(cond) < 2:
                        continue
                    newly = chosen_members(cond) - before.get(cond, set())
                    max_jump = max(max_jump, len(newly))
        assert max_jump <= 1, f"a condition gained {max_jump} newly-chosen members in one round"


def _build_level1_grid(m, base_offset, M):
    """3x3 grid of level-1 CoreItems, each itself wrapping a 3x3 grid of
    base tiles (so each level-1 covers 3x3 base-tile-indices worth, i.e. a
    3*M x 3*M cell region - M=3 giving the familiar 9x9 size used
    throughout this session's own scratch exploration) - the two-level
    structure used by update_aliases_down's own worked examples (2026-10-25:
    "constructing the core(2) from cores (0:3,0:3)"). base_offset is the
    (row, col) base-tile-index this 3x3-of-base-tiles block starts at.
    Returns a dict keyed by local (li, lj) in 0..2, each value a level-1
    CoreItem with .forced_by already computed (its own
    children's too)."""
    level1 = {}
    for li in range(3):
        for lj in range(3):
            base_sub = np.empty((3, 3), dtype=object)
            for dci in range(3):
                for dcj in range(3):
                    ci = base_offset[0] + li * 3 + dci
                    cj = base_offset[1] + lj * 3 + dcj
                    r0, c0 = ci * M, cj * M
                    core = CoreItem(m[r0:r0 + M, c0:c0 + M], margin=None,
                                     realm=((r0, r0 + M), (c0, c0 + M)))
                    Base.update_links(core)
                    base_sub[dci, dcj] = core
            r0 = (base_offset[0] + li * 3) * M
            c0 = (base_offset[1] + lj * 3) * M
            lvl1 = CoreItem(base_sub, margin=None, realm=((r0, r0 + 3 * M), (c0, c0 + 3 * M)))
            Upstream.update_links(lvl1)
            level1[(li, lj)] = lvl1
    return level1


def _reachable_via_single_links(m, N, sources):
    """Whole-board brute-force BFS over the current size-1 .forced_by
    (single-link) graph, starting from sources - an independent
    reimplementation (not reusing CoreItem's own machinery) used to cross-
    check update_aliases_down's results in the tests below."""
    forward_edges = {}
    for i in range(N):
        for j in range(N):
            for cond in m[i, j].forced_by:
                if len(cond) == 1:
                    forcer_id = next(iter(cond))[0]
                    forward_edges.setdefault(forcer_id, set()).add((i, j))
    visited = set()
    frontier = set(sources)
    while frontier:
        nxt = set()
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            for t in forward_edges.get(node, ()):
                if t not in visited:
                    nxt.add(t)
        frontier = nxt
    return visited


def test_aliases_down_matches_ivos_worked_example_1_0_to_1_2():
    """Ivo's own worked example (2026-10-25): building a level-2 supertile
    from a 3x3 grid of level-1 children (core(2) from cores (0:3,0:3)),
    with (1,0) a placement tile - is there a path from (1,0) to (1,2)?
    Verified against an independent whole-board brute-force BFS seeded from
    (1,0)'s own candidate cells: no reachable cell falls inside (1,2)'s
    realm, matching aliases_down leaving (1,2) uninformed - "I think it is
    ok not to have this information on that level" was Ivo's own answer
    about depth, not about this specific sibling-pair question, which this
    test locks in as a real (not merely empty-by-luck) negative result."""
    n, M = 9, 3
    m, N = build_swept_map(n, M)
    level1 = _build_level1_grid(m, base_offset=(0, 0), M=M)
    elements = np.empty((3, 3), dtype=object)
    for li in range(3):
        for lj in range(3):
            elements[li, lj] = level1[(li, lj)]
    level2 = CoreItem(elements, margin=None, realm=((0, N), (0, N)))
    Upstream.update_links(level2)
    aliases_down = Downstream.update_aliases_down(level2, 2)

    assert aliases_down[1, 2] == set()

    r10 = level1[(1, 0)].realm
    candidates_10 = [(i, j) for i in range(r10[0][0], r10[0][1]) for j in range(r10[1][0], r10[1][1])
                      if m[i, j].placement_status == PlacementStatus.candidate]
    assert candidates_10, "expected (1,0) to actually contain a candidate cell"
    all_reachable = _reachable_via_single_links(m, N, candidates_10)
    r12 = level1[(1, 2)].realm
    reach_in_12 = {c for c in all_reachable
                   if r12[0][0] <= c[0] < r12[0][1] and r12[1][0] <= c[1] < r12[1][1]}
    assert reach_in_12 == set()


def test_aliases_down_informs_a_genuinely_reached_sibling():
    """A non-vacuous companion to the (1,0)/(1,2) test above, where reach
    does cross a sibling boundary - found by trying a few seeds, since most
    don't produce cross-level-1 reach at all (consistent with reach staying
    mostly local throughout this whole session's findings). On an n=15
    board, level-2 centered so (1,1) (local index) contains a real
    placement core, seed=9 produces a real cross-sibling touch, reaching
    into sibling (0,0).

    Non-vacuousness is confirmed independently of CoreItem's own fields
    (via the same brute-force BFS as the test above). Checked afterward:
    every informed (origin, node) in the returned aliases_down array (a
    local 3x3 array of sets since 2026-11-07, was a per-child attribute
    that was cleared before a test could see it) starts at a candidate,
    and any .forced_by entries gained as a side effect of consumption are
    genuinely correct (reachable from their own origin per the independent
    brute-force graph) - in this particular run the touched cell turns out
    to be a dead end (no further local reach inside (0,0) once consumption
    looks for one), so .forced_by staying unchanged is itself the correct,
    verified outcome here, not a failure to detect."""
    n, M = 15, 3
    m, N = build_swept_map(n, M, seed=9)
    base_offset = (2, 2)
    level1 = _build_level1_grid(m, base_offset=base_offset, M=M)
    elements = np.empty((3, 3), dtype=object)
    for li in range(3):
        for lj in range(3):
            elements[li, lj] = level1[(li, lj)]
    r0, c0 = base_offset[0] * M, base_offset[1] * M
    level2 = CoreItem(elements, margin=None, realm=((r0, r0 + 27), (c0, c0 + 27)))
    Upstream.update_links(level2)

    # confirm, independently of CoreItem's own aliases_down, that a real
    # cross-sibling touch exists in this scenario before calling
    # update_aliases_down (which will consume and clear it)
    candidate_cells = [(i, j) for i in range(N) for j in range(N)
                        if m[i, j].placement_status == PlacementStatus.candidate]
    cross_sibling_touch_found = False
    for cand in candidate_cells:
        owner = next((t for t in level1.values() if t.realm[0][0] <= cand[0] < t.realm[0][1]
                       and t.realm[1][0] <= cand[1] < t.realm[1][1]), None)
        if owner is None:
            continue
        reachable = _reachable_via_single_links(m, N, [cand])
        for other in level1.values():
            if other is owner:
                continue
            if any(other.realm[0][0] <= c[0] < other.realm[0][1]
                   and other.realm[1][0] <= c[1] < other.realm[1][1] for c in reachable):
                cross_sibling_touch_found = True
    assert cross_sibling_touch_found, "expected at least one real cross-sibling touch in this scenario"

    before = {key: set(t.links) for key, t in level1.items()}
    aliases_down = Downstream.update_aliases_down(level2, 2)

    assert aliases_down.shape == (3, 3)
    assert any(aliases_down[idx] for idx in np.ndindex(3, 3)), "expected at least one child to be informed"
    for idx in np.ndindex(3, 3):
        for origin, _node in aliases_down[idx]:
            assert m[origin].placement_status == PlacementStatus.candidate

    for key, t in level1.items():
        for origin, node, exit_id in t.links - before[key]:
            assert m[origin].placement_status == PlacementStatus.candidate
            if node != DUMMY:
                reachable = _reachable_via_single_links(m, N, [origin])
                assert node in reachable and exit_id in reachable


def test_aliases_down_applies_the_same_way_one_level_up():
    """Ivo (2026-10-25): "Treat a nonempty list of the parent just the same
    way" - update_aliases_down must work identically regardless of which
    level it's called at, using whatever level the children's own
    .forced_by happens to be computed at. Verified with a
    small, deliberately-constructed (not swept-from-a-real-board) scenario
    so the test doesn't depend on a real sweep happening to produce reach
    that crosses a level-2 boundary: three synthetic side-by-side CoreItems
    for a, b (elements=None - plain relay/origin facts, nothing further to
    consume through) and a REAL one-child CoreItem for c, so consumption
    (added 2026-10-25, same day - "update the CoreItems .forced_by and
    clear the .aliases_down list") has real structure to extend through and
    leaves an observable .forced_by trace - .aliases_down itself is always
    cleared by the end regardless of which sibling was actually informed
    (consumption's own last step), so it can no longer distinguish
    "informed then cleared" from "never touched" on its own; .forced_by
    growth is what proves specifically c, not a or b, was the one reached.

    Restated 2026-11-07 for the "last cell inside" convention, with
    (id_in, id_out, id_exit) triples in .links: b's candidate (5,10)
    reaches b's last cell (5,17), which links into c at (5,18); c's
    grandchild continues from (5,17) to its own last cell (5,26), which
    links out of c at (5,27)."""
    Pyramid.set_placement_tiles({(1, 3)}, 3, 0)  # (5,10) is a candidate (tile rows 3-5, cols 9-11)
    a = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    b = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))
    a.links = set()
    b.links = {((5, 10), (5, 17), (5, 18))}

    # c: a real one-child CoreItem whose own grandchild already knows the
    # chain entering from (5,17) reaches its last cell (5,26), which links
    # out of c's own realm (cols 18:27) - so consumption has something
    # concrete to discover, leaving genuinely outside c, not just gc.
    gc = CoreItem(elements=None, margin=None, realm=((0, 9), (18, 27)))
    gc.links = {((5, 17), (5, 26), (5, 27))}
    c = CoreItem(np.array([[gc]], dtype=object), margin=None, realm=((0, 9), (18, 27)))
    c.links = set()

    parent = CoreItem(np.array([[a, b, c]], dtype=object), margin=None, realm=((0, 9), (0, 27)))
    aliases_down = Downstream.update_aliases_down(parent, 2)

    assert aliases_down[0, 0] == set() and a.links == set()
    assert aliases_down[0, 1] == set() and b.links == {((5, 10), (5, 17), (5, 18))}
    assert aliases_down[0, 2] == {((5, 10), (5, 17))}
    assert c.links == {((5, 10), (5, 26), (5, 27))}, \
        "expected c's own consumption to relay b's origin through to its last cell (5,26)"


def test_aliases_down_relays_through_a_siblings_own_forced_by():
    """Ivo's own follow-up example (2026-10-25): "CoreItem a has a 3x3 array
    of child CoreItems, and a .forced_by list with ([a=foreign candidate
    id], [b=some id in the realm, say in (0,0)]). Say (0,0) .forced_by has
    an entry (b, [c in (1,0)]). Then I expect a to get an entry in
    c.links [i.e. (1,0)'s own .aliases_down]." Caught a real gap: the
    first version of update_aliases_down only used children's
    candidate-started pairs to build its walk edges, so it could follow
    a's own reach INTO sibling (0,0) (since that's recorded in the sibling
    that owns 'a'), but had no way to continue the walk
    FURTHER once it reached b - a plain touched cell, not itself a
    candidate - even though (0,0)'s own .forced_by already knows b forces c
    onward into (1,0). Fixed by sourcing the walk's edges from every
    child's .forced_by, while keeping walk ORIGINS restricted to
    candidates only (is_candidate since 2026-11-07, when .placement_reach
    was removed).

    Built directly with synthetic CoreItems rather than a real board, so the
    scenario exactly matches Ivo's own description rather than depending on
    a real sweep happening to produce it. c10 (the final destination) is a
    real one-child CoreItem, not elements=None, so update_aliases_down's
    own consumption step (added 2026-10-25, same day - "update the
    CoreItems .forced_by and clear the .aliases_down list") has real
    structure to extend through and leaves an observable .forced_by trace -
    .aliases_down alone can no longer prove the relay happened, since it is
    always cleared by the end regardless.

    Restated 2026-11-07 for the "last cell inside" convention: origin_a's
    chain ends in 'other' at its last cell (5,9), linking into (0,0) at
    (5,8); (0,0) carries it from (5,9) to its own last cell (8,5), linking
    down into (1,0) at (9,5); (1,0)'s grandchild carries it from (8,5) to
    its last cell (17,2), linking out of (1,0) at (18,2). Each step is one
    (id_in, id_out, id_exit) triple in .links."""
    other = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))
    c00 = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    c11 = CoreItem(elements=None, margin=None, realm=((9, 18), (9, 18)))

    origin_a = (5, 15)       # foreign candidate, lives in 'other'
    Pyramid.set_placement_tiles({(1, 5)}, 3, 0)  # origin_a's tile (rows 3-5, cols 15-17); no other id here is a candidate
    last_in_other = (5, 9)   # other's last cell, links into (0,0)
    b_in_00 = (8, 5)         # (0,0)'s last cell on the chain, links down into (1,0)
    d_last_10 = (17, 2)      # (1,0)'s last cell on the chain, links out of (1,0) (rows 9:18)

    other.links = {(origin_a, last_in_other, (5, 8))}
    c00.links = {(last_in_other, b_in_00, (9, 5))}
    c11.links = set()

    gc10 = CoreItem(elements=None, margin=None, realm=((9, 18), (0, 9)))
    gc10.links = {(b_in_00, d_last_10, (18, 2))}
    c10 = CoreItem(np.array([[gc10]], dtype=object), margin=None, realm=((9, 18), (0, 9)))
    c10.links = set()

    elements = np.array([[other, c00], [c10, c11]], dtype=object)
    parent = CoreItem(elements, margin=None, realm=((0, 18), (0, 18)))
    aliases_down = Downstream.update_aliases_down(parent, 2)

    # elements = [[other, c00], [c10, c11]]
    assert aliases_down[0, 0] == set()
    assert aliases_down[0, 1] == {(origin_a, last_in_other), (origin_a, b_in_00)}
    assert c00.links == {(last_in_other, b_in_00, (9, 5))}
    assert aliases_down[1, 0] == {(origin_a, b_in_00)}
    assert c10.links == {(origin_a, d_last_10, (18, 2))}, \
        "expected origin_a's reach to relay through (0,0)'s forced_by into (1,0), " \
        "then through (1,0)'s own consumption to its last cell d_last_10"
    assert aliases_down[1, 1] == set() and c11.links == set()


def test_forced_by_on_realm_matches_initial_map_for_an_interior_core():
    """conditions.forced_by_on_realm (added 2026-10-31, "compute on the
    superlattice") against conditions.initial_map's own whole-board build,
    sliced to the same cells, for a core nowhere near the true board edge -
    the case where FORCED_BY_REACH=2 alone (no off-board filtering) should
    already make the two identical, since every real triple member is
    guaranteed inside the core+margin-2 window regardless of the true
    board's own size."""
    shape = (15, 15)
    whole = initial_map(shape)
    realm = ((6, 9), (6, 9))  # core (2,2) on a 3x3-of-3x3 split, far from every true edge

    actual = forced_by_on_realm(realm, shape)
    (row_start, row_end), (col_start, col_end) = realm
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            assert actual[i - row_start, j - col_start].forced_by == whole[i, j].forced_by


def test_forced_by_on_realm_matches_initial_map_for_a_corner_core():
    """Same cross-check as above, but for a core touching the true board's
    own corner (realm (0,3)x(0,3) on a 15x15 board) - the case that
    actually exercises off-board filtering (clear_of_every_edge is False
    for cells this close to row/col 0), so a mismatch here would mean
    forced_by_on_realm's own `shape`-relative on_board check is wrong, not
    just redundant."""
    shape = (15, 15)
    whole = initial_map(shape)
    realm = ((0, 3), (0, 3))

    actual = forced_by_on_realm(realm, shape)
    (row_start, row_end), (col_start, col_end) = realm
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            assert actual[i - row_start, j - col_start].forced_by == whole[i, j].forced_by


def test_collapse_synced_ids_simplifies_a_triple_and_updates_forces():
    """bruteforce.collapse_synced_ids (implemented 2026-11-01, step 1 of
    Ivo's own base-level ordering): replacing id1/id2 with id0 wherever
    they appear as .forced_by condition members collapses a triple down to
    a pair via plain frozenset dedup ("creating more links"), and
    update_forces(m) at the end keeps .forces consistent with the
    now-simplified .forced_by - without touching the CoreItem pyramid at
    all (not even built here), matching "don't recompute the tile yet"."""
    m = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            m[i, j] = SquareItem()

    id0, id1, id2, idx = (0, 0), (1, 1), (2, 2), (3, 3)
    target = (4, 4)
    m[id0].placement_status = PlacementStatus.candidate
    triple = frozenset({(id1, PlacementStatus.unknown),
                         (id2, PlacementStatus.unknown),
                         (idx, PlacementStatus.unknown)})
    m[target].forced_by = {triple}

    collapse_synced_ids(m, id0, {id1, id2})

    assert m[target].forced_by == {
        frozenset({(id0, PlacementStatus.candidate), (idx, PlacementStatus.unknown)})}
    assert any(b == target for b, _conditions in m[id0].forces)


def test_collapse_synced_ids_collapses_a_pair_entirely_into_a_single_link():
    """The further edge of the triple->pair test above: when EVERY member
    of a condition maps to the same id0 (a pair {id1, id2} with both
    syncing), the frozenset rebuild in collapse_synced_ids produces the
    identical (id0, status) tuple once per original member, so it
    collapses all the way to a size-1 condition - "(id0, id0)" becoming
    plain id0, exactly the old pipeline's own single-link fact (.links'
    own shape) - with no special-case code needed for it, same frozenset
    dedup mechanism as the partial (triple->pair) case."""
    m = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            m[i, j] = SquareItem()

    id0, id1, id2 = (0, 0), (1, 1), (2, 2)
    target = (4, 4)
    m[id0].placement_status = PlacementStatus.candidate
    pair = frozenset({(id1, PlacementStatus.unknown), (id2, PlacementStatus.unknown)})
    m[target].forced_by = {pair}

    collapse_synced_ids(m, id0, {id1, id2})

    assert m[target].forced_by == {frozenset({(id0, PlacementStatus.candidate)})}

