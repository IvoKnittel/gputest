import copy

import numpy as np

from map_of_squares import StateEnum, InvalidTilingError, set_square_chosen
from alert_graphs import (RING_OFFSETS,
                           DIAGONAL_OFFSETS,
                           iter_alert_thirds,
                           set_alert_blocked,
                           set_alert_chosen_set_links)
from representation import display_closure_step, margin_ring_positions

# Sentinel state_value both find_secondary_links5x5's and
# find_secondary_links's own local windows use for a position off the real
# board - never equal to any real StateEnum.value (0, 1, 2), so a BR-009-style
# equality check against StateEnum.free.value already excludes it for free.
_OFFBOARD = -1


def _call_step_asserts(m, asserts_single, step_name):
    """Shared implementation behind every pipeline function's own `asserts`
    parameter (find_alerts_set_links, find_secondary_links, assign_paths,
    get_blocked_links, dissolve_blocked_paths, place_square_in_seat_closed):
    look up step_name (that function's own name, matching one of
    test_utils.DoClosureSteps' member names) in DoClosureSteps, find the
    (step, fn) pair in asserts_single.asserts (a
    test_utils.DoClosureAssertsSingle) whose step matches, and call fn(m).
    Does nothing if asserts_single carries no entry for this step.

    test_utils.DoClosureSteps is imported here, locally, rather than at this
    module's own top level: test_utils already imports do_closure/
    place_square from closure.py, so importing test_utils back at closure.py's
    own module level would form a cycle. A caller can only ever reach this
    function with a real DoClosureAssertsSingle in hand (constructed via
    test_utils), which means test_utils is already fully loaded by the time
    this runs - so the local import here is always safe, just deferred.
    """
    from test_utils import DoClosureSteps
    step = DoClosureSteps[step_name]
    for candidate_step, fn in asserts_single.asserts:
        if candidate_step == step:
            fn(m)
            return


def display(m, display_single, step_name, on_error=False):
    """Shared implementation behind do_closure_intern's own `display`
    parameter - the display-mechanism equivalent of _call_step_asserts.
    Looks up step_name (a DoClosureSteps member name, or None for
    do_closure_intern's own fixed end-of-pass calls) among display_single's
    own .items (a test_utils.DoClosureDisplaySingle), and for every item
    whose .step matches step_name AND whose .on_error flag matches this
    call's own on_error argument, calls display_closure_step(m, **item.kwargs)
    (a fresh colormap=np.zeros((*m.shape, 3)) is added to kwargs here unless
    the item's own kwargs already supplies one). Returns True if any of those
    calls' own return value was truthy (display_closure_step's diagonal-
    conflict signal - see its own docstring), False otherwise (including when
    display_single is None, or nothing matched).

    on_error distinguishes do_closure_intern's two calling modes: on_error=
    False (the normal, live pass - do_closure_intern's own default,
    error_replay=False) shows only items with on_error=False, skipping every
    on_error=True item ("show only on error" - see do_closure's own
    docstring for the replay that actually shows them). on_error=True (used
    only by that replay, error_replay=True) inverts it: shows only
    on_error=True items, skipping the rest - so nothing is ever shown twice
    between the live pass and the replay.

    test_utils.DoClosureSteps is imported here, locally, for the same reason
    _call_step_asserts imports it locally - see that function's own
    docstring.

    Named exactly `display`, same as the `display` parameter do_closure_intern
    and do_closure each take - deliberate, matching how `asserts`'s own
    per-pass handling already works, though it does mean neither of those two
    functions can call this one by its bare name once their own `display`
    parameter has shadowed it; see the module-level `_display_step` alias
    just below, kept for exactly that purpose.
    """
    from test_utils import DoClosureSteps
    if display_single is None:
        return False
    step = DoClosureSteps[step_name] if step_name is not None else None
    found_error = False
    for item in display_single.items:
        if item.step != step or item.on_error != on_error:
            continue
        kwargs = dict(item.kwargs)
        kwargs.setdefault('colormap', np.zeros((*m.shape, 3)))
        if display_closure_step(m, **kwargs):
            found_error = True
    return found_error


# Alias used by do_closure_intern/do_closure to call display() above despite
# each of them also having its own parameter named `display` (which, once
# bound, shadows the module-level function of the same name for the rest of
# that function body - ordinary Python scoping, not a bug) - see display()'s
# own docstring for why the shared helper keeps that name anyway.
_display_step = display


def find_alerts_set_links(map_of_squares, asserts=None):
    """
    A seat (team term - see docs/rose_cascades_and_holes/README.md - for what this
    function's own name calls an "alert" as a noun) is a 2x2 block with three items
    blocked and one free. Uses set_alert_blocked, set_alert_chosen_set_links, and
    iter_alert_thirds to find every seat a currently-free cell threatens to
    complete, and record the promise.

    Inputs: reads .state map-wide, and .state of each free cell's 8-neighbour
    ring.

    Outputs: writes .alert_blocked (on the free cell itself) and
    .alert_chosen/.forces/.forced_by (on ring neighbours); returns None.
    asserts (a test_utils.DoClosureAssertsSingle, or None) is looked up via
    _call_step_asserts once this function's own work is done, under this
    function's own DoClosureSteps entry - see _call_step_asserts' and
    do_closure's own docstrings for why (a caller that needs do_closure's
    display but also wants to check an intermediate state uses this hook
    instead of hand-running the pipeline itself).

    Scope: local - every cell's read and write is confined to its own fixed
    1-ring neighbourhood, independent of every other cell's outcome.

    -------------------------------------------------------------------------

    For a free item, look at its 8 neighbours (direct +
    diagonal) in ring order (RING_OFFSETS). Each of the 4 possible 2x2 blocks
    touching the item is that item plus one run of 3 consecutive ring indices
    (QUADRANT_TRIPLES): two direct neighbours and the diagonal corner between them.
    If two of those three are already blocked and the third is still free, blocking
    this item would turn that block into a real seat - iter_alert_thirds yields
    exactly that free third corner. The two
    already-blocked can be either a ring-adjacent pair (direct+corner, or
    corner+direct) or the triple's two direct neighbours themselves (leaving the
    corner between them, though not ring-adjacent to either, as the free third).
    When that happens: set_alert_blocked raises .alert_blocked on the item under
    consideration, and set_alert_chosen_set_links raises .alert_chosen on the free
    completing corner, recording the pairing by adding the alert_chosen item's
    index to the alert_blocked item's .forces.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:  # BR-001
                continue
            ring = [map_of_squares[i + di, j + dj] for di, dj in RING_OFFSETS]
            set_alert_blocked(item, ring)
            if item.alert_blocked:  # BR-002
                set_alert_chosen_set_links(i, j, ring)
    if asserts is not None:
        _call_step_asserts(map_of_squares, asserts, 'find_alerts_set_links')


# find_secondary_links5x5's fixed local window - deliberately
# radius 2 (5x5, sixteen blocks), the size find_secondary_links itself used
# before test_closure_refactor.py's regression case proved radius 3 is
# required for full correctness (see find_secondary_links's own docstring).
# Kept at radius 2 here on purpose: this function is the known-incomplete,
# cheaper-window comparison point, not a second correct implementation.
_SECONDARY_WINDOW_RADIUS_5X5 = 2
_SECONDARY_WINDOW_SIZE_5X5 = 2 * _SECONDARY_WINDOW_RADIUS_5X5 + 1

_SECONDARY_BLOCK_TOP_LEFTS_5X5 = [
    (bi, bj)
    for bi in range(_SECONDARY_WINDOW_SIZE_5X5 - 1)
    for bj in range(_SECONDARY_WINDOW_SIZE_5X5 - 1)
]
_SECONDARY_BLOCK_ROWS_5X5 = np.array([[bi, bi, bi + 1, bi + 1] for bi, _ in _SECONDARY_BLOCK_TOP_LEFTS_5X5])
_SECONDARY_BLOCK_COLS_5X5 = np.array([[bj, bj + 1, bj, bj + 1] for _, bj in _SECONDARY_BLOCK_TOP_LEFTS_5X5])


def find_secondary_links5x5(map_of_squares, asserts=None):
    """Same vectorized, array-windowed approach as find_secondary_links (see
    its own docstring for the full rationale: X0/XC instead of a dict, all
    fixed blocks checked via one count_nonzero per window instead of a
    targeted scan) - but deliberately kept at the smaller, radius-2 (5x5,
    sixteen-block) window find_secondary_links itself used before its own
    regression case (test_closure_refactor.py) proved that too small.

    This is no longer a second correct implementation to check the real one
    against - it's the cheaper, known-incomplete comparison point: a block
    touching a radius-2 override position can itself extend to radius 3 (a
    block's top-left can be pos-1), so any secondary link whose completing
    corner only shows up at that outer ring is silently missed here, the
    same way find_secondary_links's own first draft missed
    test_closure_refactor.test_window_radius_must_be_3_not_2's (6, 4) ->
    (4, 3) edge. Never produces a *wrong* edge, only possibly an incomplete
    set - see test_closure_refactor.py's own coverage of exactly this
    subset relationship.

    Inputs/Outputs/Scope: identical to find_secondary_links's own docstring,
    modulo the smaller window.
    """
    rows, cols = map_of_squares.shape
    R = _SECONDARY_WINDOW_RADIUS_5X5

    def state_value(pos):
        r, c = pos
        if 0 <= r < rows and 0 <= c < cols:
            return map_of_squares[r, c].state.value
        return _OFFBOARD

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            p_item = map_of_squares[i, j]
            if not p_item.alert_blocked:  # BR-003
                continue

            ring = [map_of_squares[i + di, j + dj] for di, dj in RING_OFFSETS]
            b_positions = {(i + RING_OFFSETS[idx][0], j + RING_OFFSETS[idx][1])
                           for idx in iter_alert_thirds(ring)}
            if not b_positions:  # BR-004
                continue

            # X0: fixed 5x5 window of REAL state around P, read once per P -
            # see find_secondary_links's own docstring for why this replaces
            # a per-candidate overrides dict.
            X0 = np.array(
                [[state_value((i + dr, j + dc)) for dc in range(-R, R + 1)] for dr in range(-R, R + 1)],
                dtype=np.int8,
            )

            def to_local(pos):
                return pos[0] - i + R, pos[1] - j + R

            # All sixteen blocks' real corners, read once per P.
            real_corners_all = X0[_SECONDARY_BLOCK_ROWS_5X5, _SECONDARY_BLOCK_COLS_5X5]  # shape (16, 4)
            real_is_seat = (
                np.count_nonzero(real_corners_all == StateEnum.blocked.value, axis=1) == 3
            ) & (
                np.count_nonzero(real_corners_all == StateEnum.free.value, axis=1) == 1
            )  # BR-011, vectorized over every block at once

            for di, dj in DIAGONAL_OFFSETS:
                a_pos = (i + di, j + dj)
                if not (0 <= a_pos[0] < rows and 0 <= a_pos[1] < cols):  # BR-005
                    continue
                a_item = map_of_squares[a_pos]
                if a_item.state != StateEnum.free:  # BR-006
                    continue

                # XC: the working copy - mutated for this candidate only,
                # then discarded (next candidate copies fresh from X0).
                XC = X0.copy()

                chosen_hyp = {a_pos} | b_positions
                chosen_local = {to_local(pos) for pos in chosen_hyp}
                for lr, lc in chosen_local:
                    XC[lr, lc] = StateEnum.chosen.value

                for ci, cj in chosen_hyp:
                    for bdi, bdj in DIAGONAL_OFFSETS:
                        nlr, nlc = to_local((ci + bdi, cj + bdj))
                        if (nlr, nlc) in chosen_local:  # BR-008
                            continue
                        if X0[nlr, nlc] == StateEnum.free.value:  # BR-009 (folds BR-007 in, same as find_secondary_links)
                            XC[nlr, nlc] = StateEnum.blocked.value

                # BR-010, reframed: one vectorized read of all sixteen fixed
                # blocks' hypothetical corners, not a scan of just the ones
                # an override touched - see find_secondary_links's own
                # docstring. Unlike that function's own thirty-six blocks,
                # this window is exactly the size that misses a block whose
                # own corners land at radius 3 - see this function's own
                # docstring.
                hyp_corners_all = XC[_SECONDARY_BLOCK_ROWS_5X5, _SECONDARY_BLOCK_COLS_5X5]  # shape (16, 4)
                hyp_is_seat = (
                    np.count_nonzero(hyp_corners_all == StateEnum.blocked.value, axis=1) == 3
                ) & (
                    np.count_nonzero(hyp_corners_all == StateEnum.free.value, axis=1) == 1
                )  # BR-012, vectorized
                newly_created = hyp_is_seat & ~real_is_seat  # BR-011 + BR-012 combined

                for block_idx in np.flatnonzero(newly_created):
                    lr, lc = _SECONDARY_BLOCK_TOP_LEFTS_5X5[block_idx]
                    local_corners = [(lr, lc), (lr, lc + 1), (lr + 1, lc), (lr + 1, lc + 1)]
                    free_corner = next(pos for pos in local_corners
                                        if XC[pos] == StateEnum.free.value)
                    corner_pos = (free_corner[0] - R + i, free_corner[1] - R + j)
                    corner_item = map_of_squares[corner_pos]
                    if corner_item.forced_by & b_positions:  # BR-013
                        continue  # already reachable from a via some b in B
                    a_item.forces.add(corner_pos)
                    corner_item.forced_by.add(a_pos)
                    corner_item.alert_chosen = True
    if asserts is not None:
        _call_step_asserts(map_of_squares, asserts, 'find_secondary_links')


# find_secondary_links's fixed local window: every position its
# per-candidate simulation can ever read or write is within this radius of
# P. P's own ring (a candidates and B members alike) is radius 1; their own
# diagonal-blocking footprint (the overrides) reaches one hop further, radius
# 2; and a block *touching* a radius-2 override position can extend one cell
# past it again (a block's top-left can be pos-1), radius 3 - see the
# function's own docstring for the full argument and why that still beats a
# dict.
_SECONDARY_WINDOW_RADIUS = 3
_SECONDARY_WINDOW_SIZE = 2 * _SECONDARY_WINDOW_RADIUS + 1

# Every 2x2 block's top-left corner inside the window, in (row, col) index
# pairs into a (_SECONDARY_WINDOW_SIZE, _SECONDARY_WINDOW_SIZE) array -
# computed once at import time, identical for every P, so the per-candidate
# hot loop below never rebuilds it: it just fancy-indexes X0/XC with these
# two arrays to pull all thirty-six blocks' four corners out in one
# vectorized read. Corner order within a block matches find_secondary_links's own
# [(bi,bj), (bi,bj+1), (bi+1,bj), (bi+1,bj+1)].
_SECONDARY_BLOCK_TOP_LEFTS = [
    (bi, bj)
    for bi in range(_SECONDARY_WINDOW_SIZE - 1)
    for bj in range(_SECONDARY_WINDOW_SIZE - 1)
]
_SECONDARY_BLOCK_ROWS = np.array([[bi, bi, bi + 1, bi + 1] for bi, _ in _SECONDARY_BLOCK_TOP_LEFTS])
_SECONDARY_BLOCK_COLS = np.array([[bj, bj + 1, bj, bj + 1] for _, bj in _SECONDARY_BLOCK_TOP_LEFTS])


def find_secondary_links(map_of_squares, asserts=None):
    """Same P/a selection and BR-003..BR-013 contract as find_secondary_links5x5
    (same three fields written, same mechanism) - restructured around a
    fixed-size local array instead of a per-candidate dict, as a CPU-side
    stand-in for what a CUDA kernel over this stage would actually want to
    touch. This still runs as ordinary NumPy/Python, one P at a time -
    nothing here is on a GPU - it only changes the *shape* of the working
    data to one a GPU port could take as-is.

    This is the sole correct implementation in the codebase - find_secondary_
    links5x5 shares its mechanism but deliberately keeps the smaller,
    radius-2 window this function itself used before its own first draft's
    regression case (below) proved that too small; it is not, and is not
    meant to be, algorithmically identical to this one. Inputs/Outputs/Scope
    otherwise match find_secondary_links5x5's own docstring. test_closure_
    refactor.py verifies, board-wide, that find_secondary_links5x5's own
    .forces edges are always a *subset* of this function's - never a false
    positive, only possibly incomplete.

    -------------------------------------------------------------------------

    Why the dict has to go: overrides there is keyed by arbitrary (row, col)
    tuples, a different number of them for every (P, a) pair - no fixed
    shape, no uniform memory layout, nothing SIMD lanes can stride over in
    lockstep. X0 replaces it: a fixed 7x7 window of *real* .state, read once
    per P (not once per candidate), each cell holding a small int - the
    StateEnum's own .value, or _OFFBOARD (-1) past the real board's edge. The
    radius is 3, not the radius-2 reach of the overrides themselves: a
    radius-2 override position is still only *examined* by a block whose own
    top-left can sit one cell further out (BR-010's `for bi in (pi-1, pi)`),
    so the checked blocks' own corners can land at radius 3 even though
    nothing is ever overridden there - a real, if now-fixed, off-by-one in
    this function's own first draft (see test_closure_refactor.py's
    regression case, found via exactly this gap). XC is a working copy of
    X0, mutated in place for one candidate a's own hypothetical
    chosen/blocked cells, read to judge every block in the window, then
    thrown away and re-copied from X0 for the next candidate - "read X0
    once, copy to XC, mutate XC, use it, copy X0 back over XC, next a" is
    exactly the read/scratch/discard cycle a GPU kernel would run once per
    thread block, one block per P.

    BR-007's original bounds check has no separate counterpart here: every
    diagonal neighbour this function ever overrides is provably inside the
    7x7 window (radius 2 for the override itself, one inside the radius-3
    window it's built at), so the local index is always in range - what used
    to be an off-*board* position (negative row/col, or past rows/cols) now
    just reads back _OFFBOARD from X0, which is never equal to
    StateEnum.free.value, so BR-009's own equality check already excludes it
    for free.

    BR-010 is reframed rather than ported: instead of computing which of the
    window's 2x2 blocks an override actually touched (data-dependent - a
    different set, a different iteration count, for every candidate), all
    thirty-six fixed blocks inside the 7x7 window are checked every time.
    That is strictly more raw comparisons than the original's targeted scan,
    but a fixed iteration count with no data-dependent branching is the
    shape a SIMD lane wants - trading a little redundant work for zero
    divergence is the same trade the rest of this docstring is making
    throughout.

    What is deliberately left untouched: b_positions is still read directly
    off real SquareItem objects (same iter_alert_thirds call as find_alerts_
    set_links and find_secondary_links5x5 both already make), and
    the eventual .forces/.forced_by/.alert_chosen write still lands on real
    SquareItem objects too. Those aren't board-state arrays - they're graph
    bookkeeping with unbounded fan-out per cell (see map_of_squares.py's own
    field docs) - restructuring that is a different problem than the one
    this function's own window solves.
    """
    rows, cols = map_of_squares.shape
    R = _SECONDARY_WINDOW_RADIUS

    def state_value(pos):
        r, c = pos
        if 0 <= r < rows and 0 <= c < cols:
            return map_of_squares[r, c].state.value
        return _OFFBOARD

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            p_item = map_of_squares[i, j]
            if not p_item.alert_blocked:  # BR-003
                continue

            ring = [map_of_squares[i + di, j + dj] for di, dj in RING_OFFSETS]
            b_positions = {(i + RING_OFFSETS[idx][0], j + RING_OFFSETS[idx][1])
                           for idx in iter_alert_thirds(ring)}
            if not b_positions:  # BR-004
                continue

            # X0: fixed 7x7 window of REAL state around P, read once per P -
            # not once per candidate a, unlike the per-a overrides dict it
            # replaces. Building it is inherently one Python-level read per
            # cell (map_of_squares holds SquareItem objects, not raw state
            # values) - the array payoff below is in reusing this same X0
            # across every candidate, and in checking all thirty-six blocks
            # against it as one vectorized op instead of one at a time.
            # int8 is plenty: the only values ever stored are _OFFBOARD (-1)
            # and the three StateEnum.value's (0, 1, 2).
            X0 = np.array(
                [[state_value((i + dr, j + dc)) for dc in range(-R, R + 1)] for dr in range(-R, R + 1)],
                dtype=np.int8,
            )

            def to_local(pos):
                return pos[0] - i + R, pos[1] - j + R

            # All thirty-six blocks' real corners, read once per P via a
            # single fancy-indexed array op rather than one Python list per block.
            real_corners_all = X0[_SECONDARY_BLOCK_ROWS, _SECONDARY_BLOCK_COLS]  # shape (36, 4)
            real_is_seat = (
                np.count_nonzero(real_corners_all == StateEnum.blocked.value, axis=1) == 3
            ) & (
                np.count_nonzero(real_corners_all == StateEnum.free.value, axis=1) == 1
            )  # BR-011, vectorized over every block at once

            for di, dj in DIAGONAL_OFFSETS:
                a_pos = (i + di, j + dj)
                if not (0 <= a_pos[0] < rows and 0 <= a_pos[1] < cols):  # BR-005
                    continue
                a_item = map_of_squares[a_pos]
                if a_item.state != StateEnum.free:  # BR-006
                    continue

                # XC: the working copy - mutated for this candidate only,
                # then discarded (next candidate copies fresh from X0).
                XC = X0.copy()

                chosen_hyp = {a_pos} | b_positions
                chosen_local = {to_local(pos) for pos in chosen_hyp}
                for lr, lc in chosen_local:
                    XC[lr, lc] = StateEnum.chosen.value

                for ci, cj in chosen_hyp:
                    for bdi, bdj in DIAGONAL_OFFSETS:
                        nlr, nlc = to_local((ci + bdi, cj + bdj))
                        if (nlr, nlc) in chosen_local:  # BR-008
                            continue
                        if X0[nlr, nlc] == StateEnum.free.value:  # BR-009 (folds BR-007 in - see docstring)
                            XC[nlr, nlc] = StateEnum.blocked.value

                # BR-010, reframed: one vectorized read of all thirty-six
                # fixed blocks' hypothetical corners, not a scan of just the
                # ones an override touched - see docstring.
                hyp_corners_all = XC[_SECONDARY_BLOCK_ROWS, _SECONDARY_BLOCK_COLS]  # shape (36, 4)
                hyp_is_seat = (
                    np.count_nonzero(hyp_corners_all == StateEnum.blocked.value, axis=1) == 3
                ) & (
                    np.count_nonzero(hyp_corners_all == StateEnum.free.value, axis=1) == 1
                )  # BR-012, vectorized
                newly_created = hyp_is_seat & ~real_is_seat  # BR-011 + BR-012 combined


                for block_idx in np.flatnonzero(newly_created):
                    lr, lc = _SECONDARY_BLOCK_TOP_LEFTS[block_idx]
                    local_corners = [(lr, lc), (lr, lc + 1), (lr + 1, lc), (lr + 1, lc + 1)]
                    free_corner = next(pos for pos in local_corners
                                        if XC[pos] == StateEnum.free.value)
                    corner_pos = (free_corner[0] - R + i, free_corner[1] - R + j)
                    corner_item = map_of_squares[corner_pos]
                    if corner_item.forced_by & b_positions:  # BR-013
                        continue  # already reachable from a via some b in B
                    a_item.forces.add(corner_pos)
                    corner_item.forced_by.add(a_pos)
                    corner_item.alert_chosen = True
    if asserts is not None:
        _call_step_asserts(map_of_squares, asserts, 'find_secondary_links')


def clear_all_but_state(map_of_squares):
    """
    Clear every cell's .alert_blocked, .alert_chosen, .forces, .forced_by,
    and .path_id back to their defaults (False, False, set(), set(), set()),
    map-wide, unconditionally - .state is the only field that starts a round
    and survives it; everything else is derived fresh from .state each
    round, so nothing else should carry over. This is what lets
    find_alerts_set_links/assign_paths run from a clean slate instead of
    layering new results on top of whatever an earlier round left behind.

    Inputs: none - every field is cleared to a fixed default, regardless of
    .state or anything else already on the cell.

    Outputs: writes .alert_blocked, .alert_chosen, .forces, .forced_by, and
    .path_id on every cell, unconditionally; returns None.

    Scope: local - a pure per-cell reset, no cross-cell read at all.

    -------------------------------------------------------------------------

    find_alerts_set_links only ever adds: it skips any cell whose .state
    isn't StateEnum.free, so once a cell is placed (e.g. as part of a
    do_closure chase), whatever .alert_chosen/.alert_blocked/.forces/
    .forced_by it was carrying from an earlier round is never cleared - it just
    sits there, stale. That's silently wrong for display: colorize_with_alerts
    overlays alert_chosen/alert_blocked colour on top of the plain state colour,
    so a cell that is now genuinely chosen but still carries a stale
    alert_chosen=True renders as "promised, still free" (yellow) instead of
    actually placed (cyan) - and a stale .forces/.forced_by entry pointing at or
    from a no-longer-free cell is a dangling reference into a role that cell no
    longer plays. Needed whenever a round places more than one square at once
    (e.g. across a multi-round do_closure chase): the incremental single-step
    discipline the rest of closure.py assumes - one placement, then one
    find_alerts_set_links pass - no longer applies once several cells change
    state in the same round, so the safe thing is to recompute every cell's
    alert bookkeeping from the current board state, not just the newly-placed
    ones.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            item.alert_blocked = False
            item.alert_chosen = False
            item.forces = set()
            item.forced_by = set()
            item.path_id = set()


def check_tiling_invariant(map_of_squares):
    """
    A 2x2 block of blocked items must never happen - the
    alert_blocked/alert_chosen/forces bookkeeping find_alerts_set_links and
    assign_paths build exists specifically to prevent it.

    Inputs: reads .state of every 2x2 block of adjacent cells.

    Outputs: writes nothing; raises InvalidTilingError as a side effect, or
    returns None.

    Scope: global - each block's own check is local, but the decision to
    raise aggregates over every block on the board: one violation anywhere
    aborts the whole call, the same collapsing-to-a-single-fact shape
    get_blocked_links's return value has, just as a raise instead of a set.

    -------------------------------------------------------------------------

    If it happens anyway, raise InvalidTilingError: that signals a bug
    upstream, not a recoverable case.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = (map_of_squares[i, j], map_of_squares[i, j + 1],
                       map_of_squares[i + 1, j], map_of_squares[i + 1, j + 1])
            if all(c.state == StateEnum.blocked for c in corners):  # BR-014
                raise InvalidTilingError(f"2x2 all-blocked block at ({i}, {j})")


def place_square(map_of_squares, position):
    """
    Set position to StateEnum.chosen on map_of_squares, in place, via
    set_square_chosen (so it gets a chance to pair into a rectangle with an
    already-chosen direct neighbour), then block every diagonal neighbour of
    position that is still free. position is expected to start out free -
    e.g. a fresh cell from build_map_of_squares.

    Inputs: reads .state of each diagonal neighbour of position (to check
    it's still free before blocking it).

    Outputs: writes .state (free->chosen, plus .rectangle pairing via
    set_square_chosen) on position, and .state (free->blocked) on its
    diagonal neighbours; returns None.

    Scope: local - confined to position's own fixed 1-hop neighbourhood.

    -------------------------------------------------------------------------

    Blocks every diagonal neighbour of a placed square that is still free,
    matching the invariant map_of_squares_from_array enforces elsewhere
    (choosing an item blocks its four diagonal neighbours) - so a placed
    square's blocked neighbours show up on display_map_of_squares_3States too,
    not just the chosen square itself.

    Takes a single position, not a list: placing more than one position at
    once - without a do_closure re-evaluation between them - is exactly the
    batching discipline that let two mutually-diagonal seats both get chosen
    undetected (see get_seat_positions's docstring and
    place_square_in_seat_closed's fix). A caller that needs several positions
    placed calls this once per position, with do_closure in between when the
    positions could be mutually obligating; a caller building an unrelated,
    non-adjacent fixture (e.g. several disjoint test squares at once) simply
    loops over them directly.
    """
    set_square_chosen(map_of_squares, position)
    i, j = position
    rows, cols = map_of_squares.shape
    for di, dj in DIAGONAL_OFFSETS:
        ni, nj = i + di, j + dj
        if (0 <= ni < rows and 0 <= nj < cols
                and map_of_squares[ni, nj].state == StateEnum.free):  # BR-015
            map_of_squares[ni, nj].state = StateEnum.blocked


def add_margin_ring(map_of_squares):
    """Apply build_margin_free_map's own margin convention
    (representation.margin_ring_positions) to an existing map, in place,
    instead of building a fresh one - so whatever's already on it (e.g.
    .quality, from Quality.image_to_squares.build_quality_map) survives,
    unlike building a fresh map would. Chooses the same ring positions
    margin_ring_positions computes for this map's own shape, one at a time
    via place_square, so each one's diagonal-blocking side effect lands
    exactly as build_margin_free_map's own construction would (the ring's
    own corner-adjacent skip - see margin_ring_positions' docstring - means
    no two ring positions are ever diagonal neighbours of each other, so
    placement order among them doesn't matter).

    Inputs: reads .state of every ring position's diagonal neighbours (via
    place_square).

    Outputs: writes .state (free->chosen on the ring, free->blocked on each
    ring cell's still-free diagonal neighbours); returns map_of_squares, for
    convenience chaining the same way the outdated add_blocked_margin (see
    Quality/test_image_to_squares.py) it replaces did.

    Scope: local - each ring position's own effect is confined to its own
    fixed 1-hop diagonal neighbourhood, the same as place_square itself.
    """
    rows, cols = map_of_squares.shape
    for pos in margin_ring_positions(rows, cols):
        place_square(map_of_squares, pos)
    return map_of_squares


def get_seat_positions(map_of_squares):
    """
    Scan every 2x2 block of adjacent map_of_squares cells (same scan as
    check_tiling_invariant) for a seat - three corners blocked, one free (see
    find_alerts_set_links's docstring) - and return every seat's free corner
    found in this one scan. A pure read: does NOT place anything itself - see
    place_square_in_seat_closed for why, and for what used to happen here
    instead.

    Inputs: reads .state of every 2x2 block of adjacent cells.

    Outputs: returns a list of positions (each a seat's free corner); writes
    nothing.

    Scope: local - each block's own check reads only its own 4 cells; every
    seat found across the whole scan is collected into one list, but that
    collection is still a plain local-per-block result, not a graph walk or
    a global-identity aggregate the way get_blocked_links's return value is.

    -------------------------------------------------------------------------

    A direct state scan, independent of .alert_chosen bookkeeping - finds a
    seat wherever one currently exists on the board, not just where
    find_alerts_set_links already flagged one.

    -- Formerly here, now moved to place_square_in_seat_closed: the
    mutually-diagonal-seats gap --
    This function used to place every seat it found in one batch
    (place_square), all at once, rather than one at a time - collected
    first specifically so an earlier placement's diagonal-blocking side
    effect couldn't change a later seat's free corner out from under it
    mid-scan. That batching is exactly what let two seats found in the same
    scan, but themselves diagonal neighbours of each other, both get chosen
    at once with nothing to stop it (confirmed by direct repro - see
    Quality/test_image_to_squares.py's
    test_square_placement_random_order_supersuperlattice, and
    SquarePlacement/test_impossible.py's test_mutually_diagonal_seats/
    test_mutually_diagonal_seats_from_real_placement for a minimal one).
    Now that this function only ever returns positions rather than placing
    them, place_square_in_seat_closed places (and fully re-evaluates) one at
    a time instead - see its own docstring for how that closes the gap.
    """
    rows, cols = map_of_squares.shape
    seats = set()
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
            states = [map_of_squares[p].state for p in corners]
            if states.count(StateEnum.blocked) == 3 and states.count(StateEnum.free) == 1:  # BR-016
                seats.add(corners[states.index(StateEnum.free)])
    return list(seats)


def place_square_in_seat_closed(map_of_squares, asserts=None):
    """
    Place every seat get_seat_positions finds, to a fixed point - but one
    seat at a time, running do_closure's own full re-evaluation after each
    single placement, instead of placing every seat found in one scan
    together the way this used to. Restores, specifically for seat-filling,
    the "one placement, then one find_alerts_set_links pass" incremental
    discipline clear_all_but_state's own docstring already assumes for the
    rest of this pipeline - seat-filling used to be the one place that
    violated it.

    Inputs: none of its own - delegates entirely to get_seat_positions and
    do_closure.

    Outputs: same net effect as before (every seat filled, looped to a fixed
    point, since placing one square can block a diagonal neighbour that
    completes another 2x2 block into a fresh seat, or open up new
    consequences of its own via do_closure's own pipeline); returns a bool.
    asserts - see find_alerts_set_links's own docstring for the contract.
    Every position get_seat_positions found this pass is re-checked for
    StateEnum.free immediately before it's placed - an earlier position in
    the very same pass may already have resolved a later one (blocked it via
    diagonal side effect, or chosen it via do_closure's own cascading
    effects) - place_square would otherwise raise on an already-non-free
    cell.

    Scope: global - each individual placement is local, but do_closure
    itself is global (see its own docstring), and this function now calls
    do_closure once per seat placed, not once per whole pass.

    -------------------------------------------------------------------------

    This closes the mutually-diagonal-seats gap get_seat_positions's own
    docstring used to document as open: two seats found in the same scan
    can no longer both get chosen in one undetected batch, because they're
    no longer placed in a batch at all - the first one placed gets a full
    do_closure pass (find_alerts_set_links through check_tiling_invariant)
    before the second is ever touched. If the two seats' free corners were
    genuinely diagonal neighbours, placing the first one blocks the second's
    free corner as an ordinary diagonal-blocking side effect
    (place_square), the same way any other diagonal neighbour would be
    blocked - turning what would have been a silent diagonal-chosen conflict
    into an ordinary blocked cell instead, visible to (and, if it happens to
    complete a fully-blocked 2x2, caught by) the very next do_closure pass,
    not hidden from every check the way the old batching left it.

    Recursive: do_closure is one of this function's own callers (do_closure
    calls place_square_in_seat_closed twice, once per round), and this
    function now calls do_closure again for every single seat it places.
    Terminates regardless: every do_closure call here places at least the
    one square just given to it, strictly shrinking the board's free-cell
    count - finite and monotonic, so the recursion bottoms out even though
    it's no longer bounded by a fixed number of rounds the way a plain loop
    would be.
    """
    changed = False
    seats = get_seat_positions(map_of_squares)
    while seats:
        changed = True
        for pos in seats:
            if map_of_squares[pos].state != StateEnum.free:  # BR-037
                continue
            place_square(map_of_squares, pos)
            do_closure_intern(map_of_squares, "")
        seats = get_seat_positions(map_of_squares)
    if asserts is not None:
        _call_step_asserts(map_of_squares, asserts, 'place_square_in_seat_closed')
    return changed


def propagate_path_id_from_entries(map_of_squares):
    """
    Union every self-seeded item's path_id forward, via .forces, into
    everything it reaches.

    Inputs: reads .path_id and .forces of every cell, then walks the whole
    .forces graph reachable from any self-seeded cell.

    Outputs: writes .path_id (unions) onto every cell reached by that walk;
    returns None.

    Scope: global - an explicit BFS across .forces, unbounded in reach.

    -------------------------------------------------------------------------

    "Self-seeded" isn't a graph-structural property (not "no .forced_by") -
    it's whichever item assign_paths directly gave its own id to:
    unique_id((i, j), (rows, cols)) in item.path_id. An item that only ever received
    a foreign id through this same forward walk doesn't pass that test, so it
    never gets walked from itself - each id only needs to move forward once
    from wherever it originated.

    No .alert_chosen check anywhere in this function: once .forces/.forced_by
    exist, this works purely off them and off path_id membership - a
    non-alert_chosen pure diagonal linker that assign_paths happened to
    self-seed (because it has more than one .forces target of its own)
    propagates its id exactly the same way an alert_chosen item would.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.path_id:  # BR-018
                continue

            if not unique_id((i,j), (rows, cols)) in item.path_id:  # BR-019
                continue

            entry=item
            to_visit = list(entry.forces)
            visited = set()
            while to_visit:
                pos = to_visit.pop()
                if pos in visited:  # BR-020
                    continue
                visited.add(pos)
                item = map_of_squares[pos]
                item.path_id = item.path_id | entry.path_id
                to_visit.extend(item.forces)

def unique_id(pos, size):
    """
    Flatten pos=(i, j) into a single id, unique per cell, given
    size=(rows, cols).

    Inputs: reads nothing from the map - a pure function of its own pos/size
    arguments.

    Outputs: returns an int; writes nothing.

    Scope: local (trivially - a per-position computation, not even a map
    read).

    -------------------------------------------------------------------------

    i*M+j is only guaranteed collision-free if M >= cols
    (j never reaches M, so no row can overflow into the next one's range) -
    that held for the old i*rows+j formula only by accident, for every grid
    that happened to have rows >= cols. Multiply by whichever of rows/cols is
    the larger one: unchanged (i*rows+j) when rows >= cols, switching to
    i*cols+j only where the old formula would actually have collided (cols >
    rows - e.g. an 8x12 grid, where (0, 8) and (1, 0) both used to flatten to
    the same 8).
    """
    rows, cols = size
    if rows >= cols:  # BR-021
        return pos[0] * rows + pos[1]
    return pos[0] * cols + pos[1]

def assign_paths(map_of_squares, asserts=None):
    """
    Seed every entry, and every blocking-pair site, with its own path_id,
    then call propagate_path_id_from_entries to spread each seed forward
    along .forces.

    Inputs: reads .forces, .forced_by of every cell, plus .state and
    .forced_by of each cell's diagonal neighbours (for the self-blocking-pair
    seed).

    Outputs: writes .path_id (seeds), then calls
    propagate_path_id_from_entries (a global write - see its own header);
    returns None. asserts - see find_alerts_set_links's own docstring for
    the contract.

    Scope: global - the seeding loop here is local (one hop), but the
    function always finishes by invoking that board-wide walk, so the
    function as a whole is global.

    -------------------------------------------------------------------------

    An entry is any item with .forces but no .forced_by - no .alert_chosen
    check: a pure diagonal linker qualifies exactly like an alert_chosen item
    does, since once .forces/.forced_by exist neither this function nor
    propagate_path_id_from_entries cares how they got there.

    An entry with more than one .forces target seeds itself, as expected. An
    entry with exactly one .forces target B is pruned instead: B gets seeded
    with B's own id, not the entry's. Reasoning: "if the entry is chosen, B
    is chosen" is the entry's only possible consequence, so the entry's own
    identity adds no information about which cells have to be chosen
    together - nothing is lost letting B stand in for it. What's gained:
    when several single-target entries funnel into the same B (common), B
    ends up self-seeded once instead of the group accumulating several
    different ids that all meant the same thing.

    Separately, any item with a .forced_by (something already obligates it)
    that also has a free diagonal neighbour which is itself independently
    forced (has its own .forced_by) - a genuine diagonal-blocking pair - gets
    seeded with its own id too (added, not assigned, in case it already
    picked up an id from elsewhere in this same pass): a seed exactly like an
    entry's, just keyed off .forced_by instead of "nothing forces it", so it
    belongs here alongside the rest of the seeding.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if item.forces and not item.forced_by:  # BR-022
                if len(item.forces) == 1:  # BR-023
                    target_pos = next(iter(item.forces))
                    target = map_of_squares[target_pos[0], target_pos[1]]
                    target.path_id.add(unique_id(target_pos, (rows, cols)))
                else:  # BR-024
                    item.path_id = {unique_id((i,j), (rows, cols))}

            if item.forced_by:  # BR-025
                for di, dj in DIAGONAL_OFFSETS:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < rows and 0 <= nj < cols):  # BR-026
                        continue
                    neighbour = map_of_squares[ni, nj]
                    if neighbour.state == StateEnum.free and neighbour.forced_by:  # BR-027
                        item.path_id.add(unique_id((i, j), (rows, cols)))
                        break

    propagate_path_id_from_entries(map_of_squares)
    if asserts is not None:
        _call_step_asserts(map_of_squares, asserts, 'assign_paths')


def get_blocked_links(m, asserts=None):
    """
    Return the set of path ids flagged as self-contradicting by any cell -
    ids, not positions. Run after assign_paths, not before: path_id has to
    already be real for this to mean anything.

    Inputs: reads .path_id map-wide (plus positions, to look up each cell's
    diagonal neighbours - no other field).

    Outputs: returns a set of path ids; writes nothing to the map. asserts -
    see find_alerts_set_links's own docstring for the contract.

    Scope: global - each cell's own Q/S computation only ever looks at its
    own 4 diagonal neighbours (a local read), but the return value collapses
    every cell's local finding into one board-wide set of ids with no
    positional information attached (see "Flagged for rewrite" below) - a
    genuinely global aggregate, not a per-cell result.

    -------------------------------------------------------------------------

    For every item A that has a path_id, build Q, the union of the path_id of
    every diagonal neighbour A shares an id with. S = Q & A.path_id is every
    id that both A and one of those neighbours share. A non-empty S is a
    direct contradiction: A belongs to a path (one of the ids in S) that
    choosing A would itself break, by blocking a fellow member of that same
    path. Every id any cell's own S contributes goes into the one set this
    function returns, regardless of how many different cells separately flag
    it - so the result names which paths are contradictory, not which cells
    witnessed the contradiction (see dissolve_blocked_paths for what happens
    with that set).

    The neighbour side has no .state check, only a .path_id check - deliberate,
    not a simplification that changes what this function is documented to
    catch: A itself has no state check either (the only guard on it is
    `if not A.path_id: continue`), and diagonal adjacency is symmetric - so
    for any pair sharing an id where at least one side is free, that pair
    gets caught from whichever side is free, regardless of the other side's
    state. In every current call site, a non-empty .path_id already implies
    free anyway - clear_all_but_state clears .path_id unconditionally, and
    nothing changes any cell's .state between assign_paths and this function
    running - so this symmetry isn't actually exercised on a non-free A today.
    It costs nothing to leave the check out rather than assume that, though:
    requiring the neighbour to be free on top of its .path_id would add
    nothing correctness-wise, and would only prevent one further case (below)
    from ever being detectable here.

    -- Flagged, not resolved: a pair where *both* sides are already blocked --
    Two diagonal neighbours that are both already permanently blocked, and
    happen to share an id, are invisible to this function regardless of the
    change above - neither can ever satisfy "shares an id AND its partner is
    free", from either side. (Given clear_all_but_state's unconditional
    clear, this specific pair is also currently unreachable in practice, for
    the same reason noted above - but that's an artifact of today's call
    sequence, not a structural guarantee.) Whether that pair should count as a
    contradiction at all is open: nothing here claims two already-blocked
    cells "block each other" the way the causal "if A is chosen" story above
    does - if it matters, it's probably a separate, direct check ("does this
    path already have a blocked member anywhere on the board", no diagonal
    adjacency involved) rather than something this scan should pick up as a
    side effect. Not attempted here.

    A pure read - .path_id/.state are only ever looked at, never written, so
    this needs no snapshot-then-apply discipline of its own: nothing here
    can invalidate an earlier read.

    -- Flagged for rewrite: global id set + a second full-grid scan to match it --
    This function's whole output is a set of ids with no positional
    information; dissolve_blocked_paths then has to re-scan every cell
    (`unique_id((i, j), ...) in p`) just to find which cells those ids
    actually belong to. That round trip through unique_id/a global set
    works, but it's not in place: propagating
    the contradiction directly along each cell's own .forces/.forced_by links
    (the same links assign_paths/propagate_path_id_from_entries already walk)
    instead of round-tripping through a global id set would let get_blocked_links
    mark the origin cells itself, with no second grid-wide scan needed. Not
    done yet: noted here as a target, not attempted - the blocked_paths
    mechanism below (seed_blocked_paths/apply_blocked_paths/
    propagate_blocked_tmp_closed) is a first attempt at exactly that
    rewrite, kept separate rather than replacing this pipeline outright.

    Also flagged for rewrite: this global id vector, as get_blocked_links's
    output and dissolve_blocked_paths's input, breaks the GPU/tile computing
    model do_closure's own "Flagged for rewrite" note aims for.
    """
    rows, cols = m.shape
    p = set()
    for i in range(rows):
        for j in range(cols):
            A = m[i, j]
            if not A.path_id:  # BR-028
                continue
            Q = set()
            for di, dj in DIAGONAL_OFFSETS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols):  # BR-029
                    continue
                neighbour = m[ni, nj]
                if neighbour.path_id:  # BR-030
                    Q |= neighbour.path_id
            p |= (Q & A.path_id)
    if asserts is not None:
        _call_step_asserts(m, asserts, 'get_blocked_links')
    return p


def dissolve_blocked_paths(m, p, asserts=None):
    """Block the one cell per id in p - the cell whose own position hashes to
    that id via unique_id - and do nothing else. No eager .path_id stripping
    across the rest of the board, no .forces/.forced_by retraction.

    Motivation: that cell was, in the case that actually matters (seeded via
    assign_paths' "genuine diagonal-blocking pair" rule - .forced_by nonempty
    and a diagonal neighbour that's independently forced too), itself
    alert_chosen - the free corner of some other centre's near-seat. Blocking
    it is then exactly the same move that centre's own alert_blocked flag was
    already anticipating: one more corner of that 2x2 block goes from free to
    blocked, which - if that was the block's last free corner besides this
    one - turns it into a real seat on the spot, for do_closure's very next
    step (place_square_in_seat_closed) to find and fill, no different in
    kind from any other seat that step handles. Whatever this leaves
    dangling (this cell's own now-meaningless .forces/.forced_by, every
    other cell's now-stale copy of one of these ids in its own .path_id) is
    picked up for free by do_closure's own second round: clear_all_but_state
    wipes all of it, and the fresh find_alerts_set_links/assign_paths pass
    that follows re-derives everything from the current .state alone, in
    which this cell - no longer free - simply drops out of consideration
    entirely, the same way any other already-blocked cell does.

    Caveat this function doesn't resolve on its own: a cell seeded via
    assign_paths' OTHER rule (a plain multi-target "entry" - .forces
    nonempty, .forced_by EMPTY by definition) is not alert_chosen (alert_chosen
    is only ever set alongside a .forced_by entry) - the "this must create a
    seat" argument above doesn't apply to it. Concretely: (5, 2) in
    test_get_and_set_blocked_links_marks_blocked_tmp is documented, in that
    test's own docstring, as exactly this - no new seat borders it. Confirmed
    empirically that do_closure's second-round re-derivation fully accounts
    for it anyway: run both ways (full do_closure, this function vs. the
    original get_blocked_links/set_blocked_links pipeline it replaced) on
    that same board, the two runs end in byte-identical final states,
    including (5, 2) itself.

    Inputs: reads p (a set of ids, from get_blocked_links) against every
    cell's position via unique_id.

    Outputs: writes .state (free -> blocked) on the one cell per id in p
    whose own unique_id is a member; returns True if at least one cell was
    blocked this way, False otherwise (e.g. do_closure uses this to decide
    whether its own round actually changed anything worth displaying).
    asserts - see find_alerts_set_links's own docstring for the contract.

    Scope: local per id - unique_id is injective, so each id in p names
    exactly one cell; this is a bounded, single-write-per-id pass, not a
    board-wide aggregate.
    """
    rows, cols = m.shape
    changed = False
    for i in range(rows):
        for j in range(cols):
            if unique_id((i, j), (rows, cols)) in p:
                m[i, j].state = StateEnum.blocked
                changed = True
    if asserts is not None:
        _call_step_asserts(m, asserts, 'dissolve_blocked_paths')
    return changed


# -----------------------------------------------------------------------
# blocked_paths mechanism: an alternative to get_blocked_links/dissolve_blocked_paths,
# split into small, tile/GPU-friendly passes (see do_closure's own "Flagged for
# rewrite" note) instead of one global id set plus a second full-grid scan to
# match it. NOT wired into do_closure - it runs entirely on its own fields
# (SquareItem.blocked_paths/.is_blocked_tmp), untouched by and not touching
# the get_blocked_links/dissolve_blocked_paths pipeline do_closure actually uses.
# -----------------------------------------------------------------------

def seed_blocked_paths(m):
    """Pass 0: for every diagonally-adjacent pair of cells that share a
    path_id member, add the shared ids to each side's own .blocked_paths -
    the same per-cell computation get_blocked_links itself does (each cell's
    own Q & A.path_id), just assigned onto the cell instead of collapsed into
    one returned global set. Run once, after assign_paths - not itself looped.

    Inputs: reads .path_id of every cell and its diagonal neighbours.

    Outputs: writes .blocked_paths (assigned, not unioned - see
    apply_blocked_paths for where union applies) on every cell that has at
    least one path_id in common with a diagonal neighbour; returns None.

    Scope: local - each cell's own write depends only on its own path_id and
    its fixed 4-diagonal-neighbour ring, independent of every other cell's
    outcome (same locality as get_blocked_links's own per-cell loop).
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            A = m[i, j]
            if not A.path_id:
                continue
            Q = set()
            for di, dj in DIAGONAL_OFFSETS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                neighbour = m[ni, nj]
                if neighbour.path_id:
                    Q |= neighbour.path_id
            A.blocked_paths = Q & A.path_id


def apply_blocked_paths(m):
    """Pass 1: for every cell B seed_blocked_paths flagged (nonempty
    .blocked_paths), prune those ids out of B's own path_id, push them
    forward onto whatever B.forces (B's own consequences also inherit the
    taint), and push the blocking itself one hop backward onto whatever
    B.forced_by (the cells that would have to be chosen to force B into
    contradiction - those are the ones that are actually now impossible to
    choose). Run once - not itself looped; propagate_blocked_tmp_closed is
    what carries the backward blocking further than this one hop.

    Snapshot-then-apply: which cells qualify as B, and B's own blocked_paths/
    path_id values used in every step below, are all taken from the state at
    the start of this call - so one B's own effect on another cell's
    blocked_paths (the .forces branch) can never make that cell newly qualify
    as its own B within this same pass, and B's forced_by cells always see
    the same, single, consistent version of B's post-prune path_id (not a
    version some other B already mutated further this pass).

    Inputs: reads every cell's .blocked_paths, .path_id, .forces, .forced_by
    as they stood before this call.

    Outputs: for every B with nonempty .blocked_paths (as of entry):
    - B.path_id loses every id in B.blocked_paths;
    - every A in B.forces gains B's blocked_paths into its own (union);
    - every C in B.forced_by is set .state = StateEnum.blocked and
      .is_blocked_tmp = True (nothing is cleared - dissolve_blocked_paths
      leaves .forces/.forced_by alone too, for the same reason; here the
      graph stays intact for propagate_blocked_tmp_closed to walk further,
      and for callers/tests to still inspect);
    - every such C also has its .path_id narrowed to its intersection with
      B's own post-prune path_id (ids C had that B no longer carries are
      dropped - C's remaining membership is only ever what it still shares
      with the very cell whose contradiction is what blocked it).
    Returns None.

    Scope: local-ish - each B's own effect reaches only its fixed .forces/
    .forced_by neighbours (one hop each), not a board-wide walk; the
    snapshot discipline above is what keeps that bounded reach well-defined
    even though several B's can share a forces/forced_by neighbour.
    """
    rows, cols = m.shape
    seeded = []
    for i in range(rows):
        for j in range(cols):
            B = m[i, j]
            if B.blocked_paths:
                seeded.append((B, set(B.blocked_paths), set(B.path_id)))

    for B, blocked_paths, path_id_before in seeded:
        B.path_id -= blocked_paths
        pruned_path_id = path_id_before - blocked_paths

        for a_pos in B.forces:
            A = m[a_pos]
            A.blocked_paths |= blocked_paths

        for c_pos in B.forced_by:
            C = m[c_pos]
            C.state = StateEnum.blocked
            C.is_blocked_tmp = True
            C.path_id &= pruned_path_id


def propagate_blocked_tmp(m):
    """Pass 2, single hop: for every cell D already flagged .is_blocked_tmp
    with a still-nonempty .path_id, narrow every E in D.forced_by's own
    path_id down to its intersection with D's path_id; an E whose path_id
    actually shrinks that way *and still has something left in it* is, by
    the same reasoning apply_blocked_paths applies to its own C cells,
    itself now a genuine impossibility - blocked and flagged in turn, so a
    later pass (propagate_blocked_tmp_closed) can carry the same narrowing
    one hop further back from E.

    Both guards exist to stop a real, observed runaway cascade: without
    them, the first D whose path_id narrows all the way to empty would wipe
    every E in its .forced_by to empty too (intersecting against an empty
    set is always empty), flag every one of them blocked regardless of
    whether they ever shared anything real with D, and those newly-empty E's
    would then do the same to *their* own forced_by in the next pass -
    cascading through the whole reachable graph rather than stopping at
    genuine contradictions (confirmed on the get_and_set_blocked_links_marks_
    blocked_tmp board: 48 cells wrongly blocked instead of the correct 6). An
    empty path_id carries no specific contradiction left to push forward, so
    a D with one is skipped outright rather than treated as a source; an E
    whose intersection empties out is left untouched rather than being
    narrowed and flagged - narrowing to nothing is not itself evidence E was
    part of D's contradiction, only that this hop found no overlap.

    Inputs: reads every cell's .is_blocked_tmp, .path_id, .forced_by.

    Outputs: for every E reached this way whose .path_id changes to a
    nonempty result: .path_id is narrowed in place, .state =
    StateEnum.blocked, .is_blocked_tmp = True. Nothing is cleared (same as
    apply_blocked_paths - see its own docstring).
    Returns True if at least one cell changed this way, False otherwise.

    Scope: local per hop - each D only ever reaches its own .forced_by
    neighbours - but a D newly flagged earlier in this same scan is visible
    to a later iteration of this same pass (row-major order), so one call can
    already carry a chain more than one hop; propagate_blocked_tmp_closed's
    looping is what guarantees the rest, regardless of scan order.
    """
    rows, cols = m.shape
    changed = False
    for i in range(rows):
        for j in range(cols):
            D = m[i, j]
            if not D.is_blocked_tmp or not D.path_id:
                continue
            for e_pos in D.forced_by:
                E = m[e_pos]
                narrowed = E.path_id & D.path_id
                if narrowed != E.path_id and narrowed:
                    E.path_id = narrowed
                    E.state = StateEnum.blocked
                    E.is_blocked_tmp = True
                    changed = True
    return changed


def propagate_blocked_tmp_closed(m):
    """Run propagate_blocked_tmp to a fixed point: one D can only push the
    narrowing one hop back per pass at minimum, so keep looping until a full
    pass finds no further change - same shape as place_square_in_seat_closed
    looping get_seat_positions.

    Returns True if at least one cell was newly blocked this way, False
    otherwise.
    """
    changed = False
    while propagate_blocked_tmp(m):
        changed = True
    return changed


def has_alert_bookkeeping(m):
    """True if any cell carries .alert_blocked, .alert_chosen, or a nonempty
    .forces - i.e. find_alerts_set_links/find_secondary_links/assign_paths
    actually found or established something this pass, as opposed to a
    board with nothing free left to flag. Used by do_closure to decide
    whether its own post-assign_paths display is worth showing: those three
    functions never touch .state, so "did the map change" doesn't apply to
    them the way it does to dissolve_blocked_paths/place_square_in_seat_closed
    - this is the equivalent question for bookkeeping-only steps.

    Inputs: reads .alert_blocked, .alert_chosen, .forces of every cell.

    Outputs: returns a bool; writes nothing.

    Scope: global in principle (scans every cell), but stops at the first
    hit - typically a small, cheap prefix of the full scan on any board
    where this is actually True.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if item.alert_blocked or item.alert_chosen or item.forces:
                return True
    return False


def do_closure_intern(m, title, display=None, error_replay=False, asserts=None):
    """
    Run one full round of the closure pipeline, twice (see below for why
    twice), in place: find_alerts_set_links, assign_paths, get_blocked_links/
    dissolve_blocked_paths, place_square_in_seat_closed.

    Named do_closure_intern, not do_closure: do_closure itself is now a thin
    wrapper around this function (see its own docstring) that adds one more
    summary display, highlighting whatever changed this call, on top of
    (or instead of) this function's own per-step displays - most callers
    should call do_closure, not this one, directly.

    Inputs: none of its own - delegates entirely to the stages it calls, in
    sequence.

    Outputs: writes essentially every SquareItem field via those stages;
    returns None, or raises InvalidTilingError.

    Scope: global - includes several explicitly global stages of its own
    (assign_paths, get_blocked_links/dissolve_blocked_paths), so the pipeline
    as a whole is global regardless of how local its individual stages are.

    -------------------------------------------------------------------------

    -- Flagged for rewrite: cell-by-cell Python loops, not GPU-style tiles --
    Every stage this orchestrates (find_alerts_set_links, assign_paths,
    get_blocked_links, dissolve_blocked_paths, get_seat_positions,
    check_tiling_invariant, clear_all_but_state) is its own independent `for i in
    range(rows): for j in range(cols):` scan over every cell in plain Python.
    image_to_squares.py's insert_tile/image_squares_select_single already
    show the shape this should take instead - one "kernel call" per disjoint
    tile/core, each a batched, vectorizable operation rather than a
    scalar-per-cell Python loop. Not done yet: noted here as a target, not
    attempted - a real rewrite has to work out how each stage's cross-cell
    dependencies (e.g. assign_paths' forward walk along .forces,
    get_blocked_links' snapshot-then-apply discipline) survive being
    re-expressed over tiles instead of individual cells first.

    display (a test_utils.DoClosureDisplay, or None): .first_pass is
    forwarded as every per-stage display() call's own display_single in the
    first, displayed round below; .second_pass likewise for the second,
    silent round - mirroring exactly how `asserts` splits into .first_pass/
    .second_pass just below. Each of the module-level display() calls this
    function makes (see that function's own docstring) is still gated by the
    same condition as before this mechanism existed - that gating is
    pipeline business logic (did this stage actually find or change
    something), not something generic to any registered display item, so
    this function still computes and checks it itself:
    - after assign_paths: only if has_alert_bookkeeping(m) - find_alerts_
      set_links/find_secondary_links/assign_paths never touch .state, so
      "did the map change" doesn't apply to them; this is the equivalent
      question for a bookkeeping-only step.
    - after dissolve_blocked_paths: only if it returned True (it blocked at
      least one cell this round).
    - after place_square_in_seat_closed: only if it returned True (it placed
      at least one seat this round).
    A round that finds nothing new at some stage skips that stage's own
    display() call entirely, rather than calling it to show an unchanged
    board. Each of those three display() calls also raises InvalidTilingError
    if it finds two chosen squares that are diagonal neighbours -
    display_closure_step's show_real=True panel reports that via its own
    return value (real_space_map does not raise it directly, see its
    docstring), and display() propagates it back up (see its own docstring),
    and this is the one place that turns it back into a raise, matching
    check_tiling_invariant's already-loud handling of the other kind of
    invalid board (a fully-blocked 2x2). A caller that registers nothing for
    a given stage skips this check for that stage entirely, the same way it
    skips the display itself - a diagonal-chosen conflict can still be
    present on a run with nothing registered, just undetected by do_closure
    itself either way.

    The one call in the second, silent round that precedes find_secondary_
    links' own gated checks - "after find_secondary_links 2nd" - is NOT
    gated on any stage's own return value, same as before this mechanism
    existed: it always calls display() (though display() itself still shows
    nothing unless a caller actually registered an item for that step) -
    preserved as a pre-existing quirk, not newly introduced here.

    place_square_in_seat_closed follows dissolve_blocked_paths because a cell
    get_blocked_links flags is a genuine, permanent impossibility (see
    test_get_and_set_blocked_links_marks_blocked_tmp's (5, 2) case - blocked
    on path_id grounds alone, with no diagonal-blocking neighbour to ever
    give it away locally) - dissolve_blocked_paths already writes the real
    StateEnum.blocked immediately, not a separate pending state, so
    place_square_in_seat_closed can fill in whatever seats that
    newly-permanent blocking completes right away, no finalization step
    needed in between. Some of those same cells
    turn out to also be locally confirmed this way, but that's a bonus, not a
    requirement: the ones that aren't (like (5, 2)) are exactly the point of
    doing this at all.

    Runs the whole sequence twice: once with the optional display (so a
    caller sees the board after this round's own discoveries, before the
    next round's bookkeeping reset clears the alert/path state that produced
    them), then once more, silently, after clear_all_but_state - so that a
    round placing more than one square at once still gets a fully
    re-evaluated alert/link/path pass before it settles.

    check_tiling_invariant runs once, at the very end, after both rounds,
    raising loudly (InvalidTilingError) rather than leaving an impossible
    2x2-all-blocked board go unnoticed. Confirmed this can actually
    happen: place_square_in_seat_closed can complete
    several seats in one batch (its own scan-then-
    place-all-at-once discipline) without the per-placement re-scan that
    would otherwise catch a forming pinwheel - see the (3, 3)/(3, 4)/(4, 3)/
    (4, 4) case surfaced by test_margin_free_5x5realmap's very first round.

    This final check_tiling_invariant sits after the second, silent pass, so
    without its own safety net a failure here would raise with no display
    ever having shown the board that triggered it - the case display()'s own
    on_error item (registered under step=None, on_error=True, on whichever of
    display's two DoClosureDisplaySingle owns it - see default_display's own
    docstring for where it's registered) exists for. That item is looked up
    with on_error=error_replay, same as every other display() call this
    function makes: on a normal, live call (error_replay=False, this
    function's own default) it never matches (it's registered on_error=True),
    so nothing shows and the exception just propagates - the live pass never
    displays anything on its own failure. error_replay=True inverts every
    display() call this function makes (see display()'s own docstring): all
    of the normal, on_error=False items become unreachable, and only this one
    on_error=True item can ever match - do_closure uses this by re-running
    this function on a deepcopy of the same starting map, in error-replay
    mode, once its own real run has already raised (see do_closure's own
    docstring) - since the pipeline is a deterministic function of cell
    state, that replay reproduces the same intermediate states and the same
    eventual failure, so this on_error=True item ends up showing the exact
    board that triggered it, without the live run ever having had to display
    anything on its own critical path.

    asserts (a test_utils.DoClosureAsserts, or None): .first_pass is
    forwarded as every pipeline function's own `asserts` in the first,
    displayed round above; .second_pass likewise for the second, silent
    round - each independently None-able (skips that round's hooks only).
    See find_alerts_set_links's own docstring for what a pipeline function
    does with the DoClosureAssertsSingle it's given.
    """
    first_pass = asserts.first_pass if asserts is not None else None
    second_pass = asserts.second_pass if asserts is not None else None
    first_pass_display = display.first_pass if display is not None else None
    second_pass_display = display.second_pass if display is not None else None

    find_alerts_set_links(m, asserts=first_pass)
    find_secondary_links(m, asserts=first_pass)
    assign_paths(m, asserts=first_pass)
    if has_alert_bookkeeping(m):
        if _display_step(m, first_pass_display, 'assign_paths', on_error=error_replay):
            raise InvalidTilingError(
                    f"{title}: real_space_map found a diagonal-chosen conflict - "
                    f"see the map_of_squares panel just shown for which cells")

    blocked_something = dissolve_blocked_paths(m, get_blocked_links(m, asserts=first_pass), asserts=first_pass)
    if blocked_something:
        if _display_step(m, first_pass_display, 'dissolve_blocked_paths', on_error=error_replay):
            raise InvalidTilingError(
                f"{title}: real_space_map found a diagonal-chosen conflict - "
                f"see the map_of_squares panel just shown for which cells")

    placed_something = place_square_in_seat_closed(m, asserts=first_pass)
    if placed_something:
        if _display_step(m, first_pass_display, 'place_square_in_seat_closed', on_error=error_replay):
            raise InvalidTilingError(
                f"{title}: real_space_map found a diagonal-chosen conflict - "
                f"see the map_of_squares panel just shown for which cells")
    clear_all_but_state(m)
    find_alerts_set_links(m, asserts=second_pass)
    find_secondary_links(m, asserts=second_pass)
    # Preserved quirk (not newly introduced here - see this function's own
    # docstring): unlike the three display() calls above, this one is never
    # gated on any stage's own return value - always called, though it still
    # shows nothing unless a caller registered a find_secondary_links item.
    if _display_step(m, second_pass_display, 'find_secondary_links', on_error=error_replay):
        raise InvalidTilingError(
            f"{title}: real_space_map found a diagonal-chosen conflict - "
            f"see the map_of_squares panel just shown for which cells")
    assign_paths(m, asserts=second_pass)
    dissolve_blocked_paths(m, get_blocked_links(m, asserts=second_pass), asserts=second_pass)
    place_square_in_seat_closed(m, asserts=second_pass)
    try:
        check_tiling_invariant(m)
    except InvalidTilingError:
        # Owned by second_pass_display, not first_pass_display: this always
        # runs after the second, silent pass has fully completed, and
        # do_closure_intern has no dedicated "very end" slot of its own to
        # register it under instead - see this function's own docstring for
        # why on_error=error_replay (not a hardcoded True) is still correct
        # here despite this item itself always being registered on_error=True.
        _display_step(m, second_pass_display, None, on_error=error_replay)
        raise


def eval_map(m_before, m_after):
    """Compare two same-shape map_of_squares arrays cell by cell, and report
    every position whose .state became StateEnum.blocked, or became
    StateEnum.chosen, going from m_before to m_after.

    Inputs: reads .state of every cell in both maps.

    Outputs: returns (newly_blocked, newly_chosen) - two lists of (row, col)
    positions; writes nothing to either map.

    Scope: local - each newly_chosen position's own comparison depends only
    on that same position in both maps; newly_blocked additionally checks
    each of its own candidates against every newly_chosen position's fixed
    diagonal neighbourhood (see below) - still a bounded, fixed-radius check
    per candidate, not a graph walk.

    -------------------------------------------------------------------------

    "Became" means the position's state differs between the two maps and the
    *new* value is the state being asked about - not "was free, now X",
    though in practice those coincide here: nothing in this pipeline ever
    moves a cell away from StateEnum.chosen or StateEnum.blocked once it
    reaches one (see place_square/dissolve_blocked_paths/
    get_seat_positions), so the only real transition either list can ever
    report is free -> blocked or free -> chosen. Written as a plain
    inequality check anyway, rather than assuming that invariant holds.

    newly_blocked excludes every position that is a diagonal neighbour of
    some newly_chosen position: place_square's own diagonal-blocking side
    effect (see its docstring) blocks those unconditionally, so they're not
    an independent event worth its own yellow frame - the newly_chosen
    frame already explains them. What's left in newly_blocked is only the
    "isolated" blocks with no chosen neighbour of their own to explain them
    - i.e. dissolve_blocked_paths's own path-id-contradiction blocking (see
    its docstring), the one kind of blocking this round that a viewer
    couldn't already infer just from looking at newly_chosen.
    """
    rows, cols = m_after.shape
    newly_blocked = []
    newly_chosen = []
    for i in range(rows):
        for j in range(cols):
            before_state = m_before[i, j].state
            after_state = m_after[i, j].state
            if after_state == before_state:
                continue
            if after_state == StateEnum.blocked:
                newly_blocked.append((i, j))
            elif after_state == StateEnum.chosen:
                newly_chosen.append((i, j))

    chosen_diagonal_neighbours = {(i + di, j + dj) for i, j in newly_chosen
                                   for di, dj in DIAGONAL_OFFSETS}
    newly_blocked = [pos for pos in newly_blocked if pos not in chosen_diagonal_neighbours]
    return newly_blocked, newly_chosen


def do_closure(m, pos=None, title="", display=None, show_on_error=True, asserts=None):
    """Wrapper around do_closure_intern - what every caller should use
    instead of calling that one directly. Runs the real closure pipeline on
    m in place exactly as before (do_closure_intern is unchanged, just
    renamed), then adds one more summary display of its own: everything this
    call changed, each such cell framed in yellow, regardless of which of
    do_closure_intern's own several stages was responsible.

    pos: an optional (row, col) position to place (via place_square) before
    anything else this call does - the common "place one square, then let
    do_closure chase whatever it obligates" pattern, folded into do_closure
    itself instead of a separate place_square call before it. Called before
    m_before is captured (see below), so the placement itself is never part
    of what eval_map reports
    changed - only genuine consequences of the closure pipeline are (a
    seat that gets filled, a path that gets blocked). The placement is
    still shown, though: both of this wrapper's own display_closure_step
    calls (success and on-error) get pos forwarded as their own `pos`
    argument, framing the placed cell in red - distinct from a yellow
    newly_blocked/newly_chosen frame, which marks a consequence, not the
    placement that triggered it. None (the default) places nothing, exactly
    as before this parameter existed.

    display (a test_utils.DoClosureDisplay, or None): .first_pass/
    .second_pass are forwarded as-is to do_closure_intern's own `display`
    argument - see that function's own docstring for what it does with them.
    .top_level is this wrapper's own slot, read directly here (do_closure_
    intern never touches it) - see DoClosureDisplay's own docstring for why
    it exists as a third field alongside first_pass/second_pass rather than
    a separate parameter of its own. Each .top_level item's kwargs are
    static (built ahead of time, e.g. by test_utils.default_display), but
    pos/newly_blocked/newly_chosen are only known here, at call time - so
    rather than forcing this through the module-level display()'s own
    generic step-matching (built for do_closure_intern's fixed, statically-
    kwargs'd per-stage calls), this wrapper matches .top_level's items itself
    (by step=None and by on_error) and merges pos/newly_blocked/newly_chosen
    (and, absent an explicit title in the item's own kwargs, title or
    f"ERROR: {title}") into each match's kwargs right before calling
    display_closure_step - see _show_top_level below.

    A copy of m is taken before do_closure_intern runs (copy.deepcopy - a
    real, independent snapshot, not a view), and eval_map compares it
    against m after do_closure_intern finishes (or raises - see below) to
    get the newly_blocked/newly_chosen lists. display_closure_step then gets
    both lists as its own newly_blocked/newly_chosen arguments: if
    display.top_level carries an on_error=False item but eval_map found
    nothing changed at all (both lists empty), display_closure_step's own
    contract for that case is to display nothing, not an empty round for
    nothing - see its own docstring. No on_error=False item registered
    (e.g. test_utils.default_display(show=False, ...)) skips computing or
    showing any of this, exactly as if this wrapper's own summary-display
    logic wasn't there at all.

    If do_closure_intern raises InvalidTilingError: if show_on_error is True
    (the default), do_closure_intern is re-run once more, on a deepcopy of
    m_before (a throwaway copy - discarded once this replay returns or
    raises, never touching the real m or m_before), in error-replay mode
    (error_replay=True - see do_closure_intern's own docstring for what that
    flips) with this same display object - since the pipeline is a
    deterministic function of cell state, this reproduces the same
    intermediate states and the same eventual failure, this time actually
    surfacing whatever on_error=True items either pass's DoClosureDisplaySingle
    carries (typically just check_tiling_invariant's own failure panel - see
    default_display). This wrapper's own on-error summary display then runs
    (_show_top_level(on_error=True)) - using eval_map(m_before, m) against
    the real (failed) m, not the throwaway replay copy - before re-raising
    the original exception unchanged, so a caller of do_closure still sees
    the same failure do_closure_intern itself would have raised. show_on_error
    is the master on/off switch for this entire replay - independent of
    what's registered on either DoClosureDisplaySingle's own on_error=True
    items, show_on_error=False skips the replay outright, for callers (like
    test_margins.py) that already handle the expected-to-raise case
    themselves and would otherwise get a duplicate display.

    asserts (a test_utils.DoClosureAsserts, or None) is forwarded as-is to
    do_closure_intern's own `asserts` argument - see its own docstring for
    how .first_pass/.second_pass reach each pipeline stage.
    """
    if pos is not None:
        place_square(m, pos)
    m_before = copy.deepcopy(m)
    top_level_display = display.top_level if display is not None else None

    def _show_top_level(on_error):
        if top_level_display is None:
            return
        newly_blocked, newly_chosen = eval_map(m_before, m)
        for item in top_level_display.items:
            if item.step is not None or item.on_error != on_error:
                continue
            kwargs = dict(item.kwargs)
            kwargs.setdefault('pos', pos)
            kwargs['newly_blocked'] = newly_blocked
            kwargs['newly_chosen'] = newly_chosen
            kwargs.setdefault('colormap', np.zeros((*m.shape, 3)))
            kwargs.setdefault('title', f"ERROR: {title}" if on_error else title)
            if on_error:
                kwargs.setdefault('title_color', 'red')
            display_closure_step(m, **kwargs)

    try:
        do_closure_intern(m, title, display=display, asserts=asserts)
    except InvalidTilingError:
        if show_on_error:
            replay = copy.deepcopy(m_before)
            try:
                do_closure_intern(replay, title, display=display, error_replay=True, asserts=asserts)
            except InvalidTilingError:
                pass  # expected - the replay exists only to surface on_error=True displays, see above
            _show_top_level(on_error=True)
        raise

    _show_top_level(on_error=False)


def do_closure_old(m, pos=None, title="", display=None, show_on_error=True, asserts=None):
    """Frozen copy of do_closure, taken verbatim right before the closure
    pipeline's planned major refactor (see that refactor's own notes/commits
    for the motivating problems - among them: alerts/links only capture
    pairwise forcing, not seat-level ternary forcing like test_sudden_
    appearance.test_frozen_area's (6,7)->(8,5) link; the growing find_
    secondary_links/get_ternary_links progression; path-following being
    inherently sequential, not parallel-friendly; and wanting to know which
    cells can be chosen simultaneously for future parallel placement).

    Kept as a reference/rollback baseline - not called anywhere else in this
    codebase. Do not update this copy when do_closure itself changes; it
    exists specifically to keep showing pre-refactor behaviour. See
    do_closure's own docstring for what every parameter/the logic below
    means - unchanged here.
    """
    if pos is not None:
        place_square(m, pos)
    m_before = copy.deepcopy(m)
    top_level_display = display.top_level if display is not None else None

    def _show_top_level(on_error):
        if top_level_display is None:
            return
        newly_blocked, newly_chosen = eval_map(m_before, m)
        for item in top_level_display.items:
            if item.step is not None or item.on_error != on_error:
                continue
            kwargs = dict(item.kwargs)
            kwargs.setdefault('pos', pos)
            kwargs['newly_blocked'] = newly_blocked
            kwargs['newly_chosen'] = newly_chosen
            kwargs.setdefault('colormap', np.zeros((*m.shape, 3)))
            kwargs.setdefault('title', f"ERROR: {title}" if on_error else title)
            if on_error:
                kwargs.setdefault('title_color', 'red')
            display_closure_step(m, **kwargs)

    try:
        do_closure_intern(m, title, display=display, asserts=asserts)
    except InvalidTilingError:
        if show_on_error:
            replay = copy.deepcopy(m_before)
            try:
                do_closure_intern(replay, title, display=display, error_replay=True, asserts=asserts)
            except InvalidTilingError:
                pass  # expected - the replay exists only to surface on_error=True displays, see above
            _show_top_level(on_error=True)
        raise

    _show_top_level(on_error=False)