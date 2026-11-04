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
from cores import CoreItem
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
    """
    rows, cols = m.shape
    placement_cores = placement_core_indices(n)
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


def test_update_links_matches_direct_scan_base_and_one_level_up():
    """CoreItem.update_links, verified two ways on the 5-phase-swept map1:
    a single base tile's own .links matches a direct scan of its own 3x3
    realm exactly, and a parent built from 4 adjacent base tiles (one union
    pass) matches a direct scan of the combined 6x6 realm exactly - this is
    the original 2026-10-23 verification, now locked in as a real test
    rather than only an ad hoc script."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    def direct_scan(r0, r1, c0, c1):
        return {(i, j, next(iter(c))[0]) for i in range(r0, r1) for j in range(c0, c1)
                for c in m[i, j].forced_by if len(c) == 1}

    core = _base_tile(m, 1, 1, M)
    core.update_links()
    actual = {(cid[0], cid[1], fid) for cid, fid in core.links}
    assert actual == direct_scan(3, 6, 3, 6)

    children = np.array([[_base_tile(m, 1 + dci, 1 + dcj, M) for dcj in (0, 1)]
                          for dci in (0, 1)], dtype=object)
    for child in children.flat:
        child.update_links()
    parent = CoreItem(children, margin=None, realm=((3, 9), (3, 9)))
    parent.update_links()
    actual_parent = {(cid[0], cid[1], fid) for cid, fid in parent.links}
    assert actual_parent == direct_scan(3, 9, 3, 9)
    assert len(actual_parent) > 0, "expected at least some links in this swept scenario"


def test_forced_by_and_placement_reach_entries_never_point_inside_own_tile():
    """Every (id_in, id_out) pair recorded by a tile's own .forced_by or
    .placement_reach has id_out outside that SAME tile's own realm, for
    every one of the 49 base tiles - confirmed empirically 2026-10-25 (the
    "entries in the map are all inside the tile they belong to?" exchange):
    an entry names where reach LANDS, never territory the owning tile
    claims for itself."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    def core_of(id_):
        row, col = id_
        return row // M, col // M

    violations = []
    for ci in range(n):
        for cj in range(n):
            core = _base_tile(m, ci, cj, M)
            core.update_forced_by()
            core.update_placement_reach()
            for _id_in, id_out in core.forced_by | core.placement_reach:
                if core_of(id_out) == (ci, cj):
                    violations.append(((ci, cj), id_out))
    assert violations == []


def test_placement_reach_is_subset_of_forced_by():
    """.placement_reach is always a subset of .forced_by, for every one of
    the 49 base tiles - .forced_by's own candidate-origin case and
    .placement_reach are built from the same underlying walk, just with
    .forced_by additionally allowing plain external (non-candidate) origins
    - see CoreItem.update_forced_by's own docstring."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    checked_nonempty = False
    for ci in range(n):
        for cj in range(n):
            core = _base_tile(m, ci, cj, M)
            core.update_forced_by()
            core.update_placement_reach()
            assert core.placement_reach <= core.forced_by
            if core.placement_reach:
                checked_nonempty = True
    assert checked_nonempty, "expected at least one tile with a nonempty placement_reach"


def test_hierarchy_composition_matches_monolithic_computation():
    """The core regression test for 'follow the hierarchy up': a CoreItem
    built by composing several levels of smaller CoreItems must give
    EXACTLY the same .forced_by and .placement_reach as a single CoreItem
    built directly (base-level style) over that same bigger realm in one
    pass, for both fields, at several different realms/offsets.

    This is the exact check that caught a real bug in _compose_up's first
    version (2026-10-25): it treated every id that was ever an origin for
    ANY child as unconditionally still a valid origin at every higher
    level, which is correct for a confirmed candidate (.placement_reach)
    but wrong for a plain external-entry origin in .forced_by, since a
    non-candidate id that was merely external to one small child can become
    interior to a bigger composed realm - see squareplacement_superlattice_
    refactor memory note, 2026-10-25 entry, for the worked example and the
    require_external_origin fix in _compose_up/update_forced_by.
    """
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    base = np.empty((n, n), dtype=object)
    for ci in range(n):
        for cj in range(n):
            core = _base_tile(m, ci, cj, M)
            core.update_forced_by()
            core.update_placement_reach()
            base[ci, cj] = core

    def composed_vs_monolithic(sub, realm):
        composed = CoreItem(sub, margin=None, realm=realm)
        composed.update_placement_reach()
        composed.update_forced_by()
        (r0, r1), (c0, c1) = realm
        monolithic = CoreItem(m[r0:r1, c0:c1], margin=None, realm=realm)
        monolithic.update_placement_reach()
        monolithic.update_forced_by()
        return (composed.forced_by == monolithic.forced_by
                and composed.placement_reach == monolithic.placement_reach)

    assert composed_vs_monolithic(base[0:6, 0:6], ((0, 18), (0, 18)))
    assert composed_vs_monolithic(base[1:4, 2:5], ((1 * M, 4 * M), (2 * M, 5 * M)))
    assert composed_vs_monolithic(base[2:5, 1:4], ((2 * M, 5 * M), (1 * M, 4 * M)))
    assert composed_vs_monolithic(base[0:7, 0:7], ((0, N), (0, N)))


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
    CoreItem with .forced_by/.placement_reach already computed (its own
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
                    core.update_forced_by()
                    core.update_placement_reach()
                    base_sub[dci, dcj] = core
            r0 = (base_offset[0] + li * 3) * M
            c0 = (base_offset[1] + lj * 3) * M
            lvl1 = CoreItem(base_sub, margin=None, realm=((r0, r0 + 3 * M), (c0, c0 + 3 * M)))
            lvl1.update_forced_by()
            lvl1.update_placement_reach()
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
    level2.update_forced_by()
    level2.update_placement_reach()
    level2.update_aliases_down()

    assert level1[(1, 2)].aliases_down == set()

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
    (via the same brute-force BFS as the test above), not by reading
    .aliases_down after the call - update_aliases_down now consumes and
    clears every informed child's own .aliases_down as its own last step
    (2026-10-25: "At the end, you should update the CoreItems .forced_by
    and clear the .aliases_down list"), so .aliases_down is always empty
    by the time this test can observe it, whether or not anything happened.
    Checked instead: .aliases_down really is empty everywhere afterward
    (the clearing itself), and any .forced_by entries .forced_by gained as
    a side effect of consumption are genuinely correct (reachable from
    their own origin per the independent brute-force graph) - in this
    particular run the touched cell turns out to be a dead end (no further
    local reach inside (0,0) once consumption looks for one), so .forced_by
    staying unchanged is itself the correct, verified outcome here, not a
    failure to detect."""
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
    level2.update_forced_by()
    level2.update_placement_reach()

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

    before = {key: set(t.forced_by) for key, t in level1.items()}
    level2.update_aliases_down()

    for t in level1.values():
        assert t.aliases_down == set()

    for key, t in level1.items():
        for origin, node in t.forced_by - before[key]:
            assert m[origin].placement_status == PlacementStatus.candidate
            assert node in _reachable_via_single_links(m, N, [origin])


def test_aliases_down_applies_the_same_way_one_level_up():
    """Ivo (2026-10-25): "Treat a nonempty list of the parent just the same
    way" - update_aliases_down must work identically regardless of which
    level it's called at, using whatever level the children's own
    .placement_reach/.forced_by happen to be computed at. Verified with a
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
    growth is what proves specifically c, not a or b, was the one reached."""
    a = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    b = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))
    a.placement_reach = set()
    a.forced_by = set()
    b.placement_reach = {((5, 10), (5, 20))}
    b.forced_by = {((5, 10), (5, 20))}

    # c: a real one-child CoreItem whose own grandchild already knows
    # (5,20) [the cell b's reach touches c at] forces (5,30), external to
    # c's own realm (cols 18:27) - so consumption has something concrete to
    # discover, landing genuinely outside c, not just outside gc.
    gc = CoreItem(elements=None, margin=None, realm=((0, 9), (18, 27)))
    gc.forced_by = {((5, 20), (5, 30))}
    gc.placement_reach = set()
    c = CoreItem(np.array([[gc]], dtype=object), margin=None, realm=((0, 9), (18, 27)))
    c.placement_reach = set()
    c.forced_by = set()

    parent = CoreItem(np.array([[a, b, c]], dtype=object), margin=None, realm=((0, 9), (0, 27)))
    parent.update_aliases_down()

    assert a.aliases_down == set() and a.forced_by == set()
    assert b.aliases_down == set() and b.forced_by == {((5, 10), (5, 20))}
    assert c.aliases_down == set()
    assert c.forced_by == {((5, 10), (5, 30))}, \
        "expected c's own consumption to relay b's origin through to (5,25)"


def test_aliases_down_relays_through_a_siblings_own_forced_by():
    """Ivo's own follow-up example (2026-10-25): "CoreItem a has a 3x3 array
    of child CoreItems, and a .forced_by list with ([a=foreign candidate
    id], [b=some id in the realm, say in (0,0)]). Say (0,0) .forced_by has
    an entry (b, [c in (1,0)]). Then I expect a to get an entry in
    c.forced_by [i.e. (1,0)'s own .aliases_down]." Caught a real gap: the
    first version of update_aliases_down only used children's
    .placement_reach to build its walk edges, so it could follow a's own
    reach INTO sibling (0,0) (since that's recorded in the sibling that
    owns 'a', via .placement_reach), but had no way to continue the walk
    FURTHER once it reached b - a plain touched cell, not itself a
    candidate - even though (0,0)'s own .forced_by already knows b forces c
    onward into (1,0). Fixed by sourcing the walk's edges from every
    child's .forced_by (which .placement_reach is always a subset of, not
    a different set), while keeping walk ORIGINS restricted to confirmed
    candidates (.placement_reach keys) only.

    Built directly with synthetic CoreItems rather than a real board, so the
    scenario exactly matches Ivo's own description rather than depending on
    a real sweep happening to produce it. c10 (the final destination) is a
    real one-child CoreItem, not elements=None, so update_aliases_down's
    own consumption step (added 2026-10-25, same day - "update the
    CoreItems .forced_by and clear the .aliases_down list") has real
    structure to extend through and leaves an observable .forced_by trace -
    .aliases_down alone can no longer prove the relay happened, since it is
    always cleared by the end regardless."""
    other = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))
    c00 = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    c11 = CoreItem(elements=None, margin=None, realm=((9, 18), (9, 18)))

    origin_a = (5, 15)       # foreign candidate, lives in 'other'
    b_in_00 = (5, 5)         # touched cell inside (0,0)'s own realm
    c_in_10 = (12, 5)        # external to (0,0), lands inside (1,0)
    d_beyond_10 = (20, 2)    # external to (1,0) (realm rows 9:18) - (1,0)'s own further reach from c_in_10

    other.placement_reach = {(origin_a, b_in_00)}
    other.forced_by = {(origin_a, b_in_00)}
    c00.placement_reach = set()
    c00.forced_by = {(b_in_00, c_in_10)}
    c11.placement_reach = set()
    c11.forced_by = set()

    gc10 = CoreItem(elements=None, margin=None, realm=((9, 18), (0, 9)))
    gc10.forced_by = {(c_in_10, d_beyond_10)}
    gc10.placement_reach = set()
    c10 = CoreItem(np.array([[gc10]], dtype=object), margin=None, realm=((9, 18), (0, 9)))
    c10.placement_reach = set()
    c10.forced_by = set()

    elements = np.array([[other, c00], [c10, c11]], dtype=object)
    parent = CoreItem(elements, margin=None, realm=((0, 18), (0, 18)))
    parent.update_aliases_down()

    assert c00.aliases_down == set() and c00.forced_by == {(b_in_00, c_in_10)}
    assert c10.aliases_down == set()
    assert c10.forced_by == {(origin_a, d_beyond_10)}, \
        "expected origin_a's reach to relay through (0,0)'s forced_by into (1,0), " \
        "then through (1,0)'s own consumption to d_beyond_10"
    assert c11.aliases_down == set() and c11.forced_by == set()


def _tile_scoped_blocked_brute_force(m, ci, cj, M):
    """Independent reimplementation of update_blocked's own base-level
    logic, scoped (like update_blocked itself, for now) to exactly one
    tile's own cells' .forced_by/.forces - not the whole board. Used to
    cross-check update_blocked without testing its own code against
    itself."""
    r0, c0 = ci * M, cj * M
    interior_ids = {(r0 + di, c0 + dj) for di in range(M) for dj in range(M)}
    forward_edges = {}
    status_of = {}
    for cid in interior_ids:
        item = m[cid]
        status_of[cid] = item.placement_status
        for cond in item.forced_by:
            if len(cond) == 1:
                fid, fstatus = next(iter(cond))
                forward_edges.setdefault(fid, set()).add(cid)
                status_of.setdefault(fid, fstatus)
    exit_edges = {}
    for cid in interior_ids:
        for b, conds in m[cid].forces:
            if b in interior_ids:
                continue
            if any(len(c) == 1 for c in conds):
                exit_edges.setdefault(cid, set()).add(b)

    def full_reach(origin):
        frontier = {origin}
        visited = set()
        reached = {origin}
        while frontier:
            nxt = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                reached.add(node)
                reached |= exit_edges.get(node, set())
                for t in forward_edges.get(node, ()):
                    if t in interior_ids and t not in visited:
                        nxt.add(t)
            frontier = nxt
        return reached

    def is_diag(a, b):
        return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1

    candidates = [cid for cid in interior_ids if status_of.get(cid) == PlacementStatus.candidate]
    expected = set()
    for id1 in candidates:
        reach1 = full_reach(id1)
        for id2 in candidates:
            if any(is_diag(x, id2) for x in reach1):
                expected.add((id1, id2) if id1 <= id2 else (id2, id1))
    return expected


def test_update_blocked_matches_tile_scoped_brute_force():
    """Ivo (2026-10-28): .blocked is a set of (id1, id2) pairs excluding
    each other - id1's own forcing chain reaches a cell diagonally adjacent
    to id2, so id1 and id2 can never both be chosen. Verified against an
    independent, tile-scoped brute-force reimplementation (not the whole
    board's global reach, which update_blocked does not claim to capture
    yet - a first, GLOBAL-scoped cross-check attempt found a real
    discrepancy that turned out to be the brute-force script's own fault,
    not a bug in update_blocked: it used the whole board's forward_edges,
    an apples-to-oranges comparison against update_blocked's explicitly
    tile-scoped definition) - for all 4 placement tiles on the 5-phase-swept
    map1."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    for ci, cj in placement_core_indices(n):
        core = _base_tile(m, ci, cj, M)
        core.update_blocked()
        expected = _tile_scoped_blocked_brute_force(m, ci, cj, M)
        assert core.blocked == expected, f"mismatch at tile ({ci},{cj})"


def test_update_blocked_pairs_are_canonical_and_within_this_tiles_candidates():
    """Every .blocked pair is stored low-id-first (same convention as
    .conflicts - see this module's own docstring) and both ids are among
    this tile's own candidate cells - update_blocked's own documented
    scope (cross-tile exclusion is not yet computed)."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    for ci, cj in placement_core_indices(n):
        r0, c0 = ci * M, cj * M
        candidates = {(r0 + di, c0 + dj) for di in range(M) for dj in range(M)
                      if m[r0 + di, c0 + dj].placement_status == PlacementStatus.candidate}
        core = _base_tile(m, ci, cj, M)
        core.update_blocked()
        for id1, id2 in core.blocked:
            assert id1 <= id2
            assert id1 in candidates and id2 in candidates


def test_update_blocked_finds_a_genuine_self_blocking_case():
    """Confirms Ivo's own "self-blocking included" case actually occurs,
    not just in theory: on the 5-phase-swept map1, tile (1,1)'s own
    candidate (3,5) self-blocks - its own forcing chain reaches a cell
    diagonally adjacent to itself, a genuine contradiction (it could never
    validly have been chosen). Cross-checked directly against the chain:
    the diagonal neighbor it reaches is found explicitly, not just asserted
    to exist."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)
    core = _base_tile(m, 1, 1, M)
    core.update_blocked()
    assert ((3, 5), (3, 5)) in core.blocked

    expected = _tile_scoped_blocked_brute_force(m, 1, 1, M)
    reach = set()
    # reconstruct (3,5)'s own full reach the same way update_blocked does,
    # to directly exhibit which diagonal neighbor causes the contradiction
    interior_ids = {(3 + di, 3 + dj) for di in range(M) for dj in range(M)}
    forward_edges = {}
    for cid in interior_ids:
        for cond in m[cid].forced_by:
            if len(cond) == 1:
                fid, _ = next(iter(cond))
                forward_edges.setdefault(fid, set()).add(cid)
    exit_edges = {}
    for cid in interior_ids:
        for b, conds in m[cid].forces:
            if b in interior_ids:
                continue
            if any(len(c) == 1 for c in conds):
                exit_edges.setdefault(cid, set()).add(b)
    frontier = {(3, 5)}
    visited = set()
    while frontier:
        nxt = set()
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            reach.add(node)
            reach |= exit_edges.get(node, set())
            for t in forward_edges.get(node, ()):
                if t in interior_ids and t not in visited:
                    nxt.add(t)
        frontier = nxt

    diagonal_hits = [x for x in reach if abs(x[0] - 3) == 1 and abs(x[1] - 5) == 1]
    assert diagonal_hits, "expected to find the specific cell causing (3,5)'s own self-block"
    assert ((3, 5), (3, 5)) in expected


def test_unblock_path_resolves_a_real_self_blocking_candidate():
    """Ivo (2026-10-29): "to unblock a path, call a new function
    unblock_path. It places the blocked cell and recomputes the tile from
    .state. Check again to conversion. Then proceed to compute
    CoreItem.forced_by." The new architecture's own analog of the old
    pipeline's dissolve_blocked_paths + do_closure_intern's second-pass
    recompute (see squareplacement_superlattice_refactor memory note,
    2026-10-29 entries) - locks in the real self-blocking case already
    found on the 5-phase-swept map1: tile (1,1)'s own candidate (3,5).

    Checks all three things Ivo asked for, in order: (1) unblock_path
    actually blocks the cell; (2) re-checking .blocked afterward ("check
    again to conversion") finds no self-blocking left anywhere, not just at
    (3,5) itself - this is a one-shot check, not a loop, since this
    particular scenario happens to converge in a single application, but
    the check covers every placement tile, not just the one (3,5) belongs
    to, in case blocking (3,5) revealed a self-block elsewhere; (3)
    CoreItem.forced_by, recomputed after, no longer has (3,5) as an origin
    anywhere - compute_conditions's own existing "drop a condition with any
    blocked member" rule removes every trace of (3,5) as a valid condition
    member board-wide, with no special-casing needed in update_forced_by
    itself."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    core = _base_tile(m, 1, 1, M)
    core.update_blocked()
    self_blocking = {id1 for id1, id2 in core.blocked if id1 == id2}
    assert self_blocking == {(3, 5)}, "expected the known (3,5) self-blocking case"

    core.update_forced_by()
    before_origins = {a for a, _b in core.forced_by if a == (3, 5)}
    assert before_origins, "expected (3,5) to have real forced_by entries before unblock_path"

    assert m[3, 5].state == StateEnum.free
    unblock_path(m, (3, 5))
    assert m[3, 5].state == StateEnum.blocked

    for ci, cj in placement_core_indices(n):
        tile = _base_tile(m, ci, cj, M)
        tile.update_blocked()
        remaining_self_blocking = {id1 for id1, id2 in tile.blocked if id1 == id2}
        assert remaining_self_blocking == set(), \
            f"expected no self-blocking left at tile ({ci},{cj}) after unblock_path"

    core_after = _base_tile(m, 1, 1, M)
    core_after.update_forced_by()
    after_origins = {a for a, _b in core_after.forced_by if a == (3, 5)}
    assert after_origins == set(), \
        "expected (3,5) to no longer appear as a forced_by origin after unblock_path"


def test_update_blocked_recursive_finds_transitive_self_blocking():
    """Ivo (2026-10-29): "if an item A from the 3x3 child array has (id1,
    id2) blocking each other and id0 from item A forces both items, id0 is
    blocked." A second, transitive way a candidate can be self-blocking,
    discovered one level up the hierarchy rather than via direct diagonal
    adjacency: if some id0's own forcing chain (composed across every
    child's own .forced_by) reaches BOTH sides of a child's own known
    .blocked pair (id1, id2) - which already means id1 and id2 can never
    both be chosen - then id0 itself could never validly be chosen either.

    Built directly with synthetic CoreItems (elements=None for the
    children, since update_blocked's own recursive branch only ever reads
    their .blocked/.forced_by/.placement_reach, never .elements) rather
    than depending on a real sweep happening to produce this specific
    configuration."""
    child_a = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    child_b = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))

    id1, id2 = (3, 3), (6, 6)
    id0 = (5, 10)

    child_a.blocked = {(id1, id2)}
    child_a.forced_by = {(id1, id2)}  # lets the combined graph continue id1 -> id2
    child_a.placement_reach = set()

    child_b.blocked = set()
    child_b.forced_by = {(id0, id1)}
    child_b.placement_reach = {(id0, id1)}  # id0 is a confirmed candidate

    elements = np.array([[child_a, child_b]], dtype=object)
    parent = CoreItem(elements, margin=None, realm=((0, 9), (0, 18)))
    parent.update_blocked()

    assert (id1, id2) in parent.blocked, "the original child pair should survive the union"
    assert (id0, id0) in parent.blocked, "id0 should be discovered self-blocking"


def test_update_blocked_recursive_matches_tile_scoped_brute_force():
    """Cross-checks the recursive update_blocked branch on real board data -
    a 3x3 group of base tiles (rows/cols 0-8) on the 5-phase-swept map1,
    which includes the known self-blocking tile (1,1) - against an
    independent brute-force reimplementation scoped to exactly the same 9
    tiles' own cells (not the whole board), same methodology as the
    base-level cross-check."""
    n, M = 7, 3
    m, N = build_swept_map(n, M)

    base = {}
    for ci in range(3):
        for cj in range(3):
            core = _base_tile(m, ci, cj, M)
            core.update_forced_by()
            core.update_placement_reach()
            core.update_blocked()
            base[(ci, cj)] = core

    elements = np.empty((3, 3), dtype=object)
    for ci in range(3):
        for cj in range(3):
            elements[ci, cj] = base[(ci, cj)]
    parent = CoreItem(elements, margin=None, realm=((0, 9), (0, 9)))
    parent.update_blocked()

    r0, r1, c0, c1 = 0, 9, 0, 9
    forward_edges = {}
    for i in range(r0, r1):
        for j in range(c0, c1):
            for cond in m[i, j].forced_by:
                if len(cond) == 1:
                    fid, _fstatus = next(iter(cond))
                    forward_edges.setdefault(fid, set()).add((i, j))

    def in_realm(x):
        return r0 <= x[0] < r1 and c0 <= x[1] < c1

    def full_reach(origin):
        frontier = {origin}
        visited = set()
        reached = {origin}
        while frontier:
            nxt = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                reached.add(node)
                for t in forward_edges.get(node, ()):
                    reached.add(t)
                    if in_realm(t) and t not in visited:
                        nxt.add(t)
            frontier = nxt
        return reached

    candidates = [(i, j) for i in range(r0, r1) for j in range(c0, c1)
                  if m[i, j].placement_status == PlacementStatus.candidate]

    child_blocked_pairs = set()
    for core in base.values():
        child_blocked_pairs |= core.blocked

    expected = set(child_blocked_pairs)
    for id0 in candidates:
        reach0 = full_reach(id0)
        for id1, id2 in child_blocked_pairs:
            if id1 in reach0 and id2 in reach0:
                expected.add((id0, id0))

    assert parent.blocked == expected


def test_push_down_leaves_a_doubly_external_forced_by_pair_where_it_already_is():
    """push_down's rule 2 (see its own docstring, corrected 2026-10-31): a
    .forced_by pair that in_realm can't route to any of this core's own
    children (both ids external even to every child) is never actually
    ownerless - _compose_up's own first-hop short-circuit (see its own
    docstring) guarantees such a pair already exists, unchanged, as a
    literal member of exactly one child's own .forced_by before this
    core's own union ever ran, and push_down can never later push anything
    into a position where that stops holding (see the docstring's own
    proof) - so this is structurally impossible to get wrong, not merely
    unobserved; no special-case code is needed or present for it. Reproduces
    the concrete 2026-10-31 example (((6,9),(9,6)), traced to child (2,2))
    with synthetic CoreItems rather than a real sweep, so the scenario is
    exact rather than incidental - push_down must leave the pair exactly
    where it already was (not copy it anywhere else, not raise)."""
    children = {}
    for ci in range(3):
        for cj in range(3):
            r0, c0 = ci * 3, cj * 3
            child = CoreItem(elements=None, margin=None,
                              realm=((r0, r0 + 3), (c0, c0 + 3)))
            child.forced_by = set()
            child.blocked = set()
            children[(ci, cj)] = child

    doubly_external = ((6, 9), (9, 6))
    children[(2, 2)].forced_by = {doubly_external}

    elements = np.empty((3, 3), dtype=object)
    for ci in range(3):
        for cj in range(3):
            elements[ci, cj] = children[(ci, cj)]
    parent = CoreItem(elements, margin=None, realm=((0, 9), (0, 9)))
    # inherited verbatim into the parent, exactly as _compose_up would leave it
    parent.forced_by = {doubly_external}
    parent.blocked = set()

    parent.push_down()  # must not raise - the pair is already homed in (2,2)

    for (ci, cj), child in children.items():
        if (ci, cj) == (2, 2):
            assert child.forced_by == {doubly_external}
        else:
            assert doubly_external not in child.forced_by


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

