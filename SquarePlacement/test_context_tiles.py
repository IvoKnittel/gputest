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

from conditions import initial_map
from closure import place_square
from bruteforce import compute_conditions, retag_forced_by, prune_same_core_candidates
from cores import CoreItem
from map_of_squares import StateEnum, PlacementStatus


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
    placement core, seed=9 produces 3 aliases_down entries, all informing
    sibling (0,0) - cross-checked against the same independent brute-force
    BFS as the test above, confirming every entry's id_out really is
    reachable from its own origin, and that the origin really is a
    candidate cell."""
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
    level2.update_aliases_down()

    total = sum(len(t.aliases_down) for t in level1.values())
    assert total > 0, "expected at least one sibling to actually be informed in this run"

    for child in level1.values():
        for origin, node in child.aliases_down:
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
    (elements=None - update_aliases_down only ever reads .placement_reach/
    .forced_by and .realm, never .elements, for its own non-base branch)
    with hand-set .placement_reach/.forced_by (the pair set on both,
    respecting the real .placement_reach <= .forced_by invariant - edges
    are sourced from .forced_by, origins from .placement_reach, see
    update_aliases_down's own docstring), composed into one parent. Only
    the sibling whose realm the pair's id_out actually falls into gets
    informed."""
    a = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    b = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))
    c = CoreItem(elements=None, margin=None, realm=((0, 9), (18, 27)))
    a.placement_reach = set()
    a.forced_by = set()
    b.placement_reach = {((5, 10), (5, 20))}
    b.forced_by = {((5, 10), (5, 20))}
    c.placement_reach = set()
    c.forced_by = set()

    parent = CoreItem(np.array([[a, b, c]], dtype=object), margin=None, realm=((0, 9), (0, 27)))
    parent.update_aliases_down()

    assert a.aliases_down == set()
    assert b.aliases_down == set()
    assert c.aliases_down == {((5, 10), (5, 20))}


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

    Built directly with synthetic CoreItems (elements=None) rather than a
    real board, so the scenario exactly matches Ivo's own description
    rather than depending on a real sweep happening to produce it."""
    other = CoreItem(elements=None, margin=None, realm=((0, 9), (9, 18)))
    c00 = CoreItem(elements=None, margin=None, realm=((0, 9), (0, 9)))
    c10 = CoreItem(elements=None, margin=None, realm=((9, 18), (0, 9)))
    c11 = CoreItem(elements=None, margin=None, realm=((9, 18), (9, 18)))

    origin_a = (5, 15)       # foreign candidate, lives in 'other'
    b_in_00 = (5, 5)         # touched cell inside (0,0)'s own realm
    c_in_10 = (12, 5)        # external to (0,0), lands inside (1,0)

    other.placement_reach = {(origin_a, b_in_00)}
    other.forced_by = {(origin_a, b_in_00)}
    c00.placement_reach = set()
    c00.forced_by = {(b_in_00, c_in_10)}
    c10.placement_reach = set()
    c10.forced_by = set()
    c11.placement_reach = set()
    c11.forced_by = set()

    elements = np.array([[other, c00], [c10, c11]], dtype=object)
    parent = CoreItem(elements, margin=None, realm=((0, 18), (0, 18)))
    parent.update_aliases_down()

    assert c00.aliases_down == {(origin_a, b_in_00)}
    assert c10.aliases_down == {(origin_a, c_in_10)}, \
        "expected origin_a's reach to relay through (0,0)'s own forced_by into (1,0)"
    assert c11.aliases_down == set()
