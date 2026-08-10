import numpy as np

from map_of_squares import StateEnum, InvalidTilingError, set_square_chosen
from alert_graphs import (RING_OFFSETS,
                           DIAGONAL_OFFSETS,
                           set_alert_blocked,
                           set_alert_chosen)
from representation import display_closure_step

def find_alerts(map_of_squares):
    """Stage 1 of the closure pipeline (set_alert_blocked, set_alert_chosen,
    iter_alert_thirds): a seat (team term - see docs/rose_cascades_and_holes/README.md
    - for what this function's own name calls an "alert" as a noun) is a 2x2 block with three items
    blocked and one free. For a free item, look at its 8 neighbours (direct +
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
    consideration, and set_alert_chosen raises .alert_chosen on the free completing
    corner, recording the pairing by adding the alert_chosen item's index to the
    alert_blocked item's .forces.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            ring = [map_of_squares[i + di, j + dj] for di, dj in RING_OFFSETS]
            set_alert_blocked(item, ring)
            if item.alert_blocked:
                set_alert_chosen(i, j, ring)


def link_patches(map_of_squares):
    """Stage 2 of the closure pipeline - formerly a relay: for every free item,
    adopt whatever a diagonal neighbour's own .forces relayed, on the premise
    that an alert_blocked neighbour's .forces still described "what to do if I
    get blocked". That premise stopped holding once set_alert_chosen started
    linking every free diagonal neighbour of an alert_blocked centre directly
    (see its docstring) - an alert_blocked neighbour's .forces describes
    something unrelated (whatever *it* separately forces as someone else's
    diagonal linker), so relaying it here actively overwrote correct stage-1
    forces with wrong ones. There is nothing left for a relay stage to find;
    this function is now a guaranteed no-op, kept only so every existing
    caller (every test's pipeline) keeps working unchanged.
    """


def mark_self_blocking_same_patch(map_of_squares):
    """First stage of remove_blocked_links: find pairs of alert_chosen items that
    diagonally block each other while sharing the same patch_id - choosing either
    one blocks the other, so the patch can never actually be chosen as a whole.

    Requires patch_id to already have converged (resolve_cycles_and_centrality) -
    before that every item's patch_id is still -1, so this returns [].

    Returns a list of ((row, col), (row, col)) pairs, each unordered pair reported
    once - propagate_deletions_once resolves each pair in turn.
    """
    rows, cols = map_of_squares.shape
    seen = set()
    pairs = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.patch_id != -1):
                continue
            for di, dj in DIAGONAL_OFFSETS:
                ni, nj = i + di, j + dj
                neighbour = map_of_squares[ni, nj]
                if neighbour.alert_chosen and neighbour.patch_id == item.patch_id:
                    key = frozenset({(i, j), (ni, nj)})
                    if key not in seen:
                        seen.add(key)
                        pairs.append(((i, j), (ni, nj)))
    return pairs


def clog_item(map_of_squares, pos):
    """Permanently exclude the item at pos - used on a common forcer of a
    mutually-blocking pair (see propagate_deletions_once), which can itself never
    be fulfilled either. Set to StateEnum.blocked, not free - it never gets chosen
    - with every other field reset to its default except .patch_id (left
    untouched - a clogged item stays on the record as having belonged to this
    patch), and its own index retracted from every other item's
    .forces/.forced_by so nothing downstream still counts on it.
    """
    item = map_of_squares[pos]
    for target in item.forces:
        map_of_squares[target].forced_by.discard(pos)
    for source in item.forced_by:
        map_of_squares[source].forces.discard(pos)
    item.state = StateEnum.blocked
    item.alert_chosen = False
    item.alert_blocked = False
    item.forces = set()
    item.forced_by = set()
    item.centrality = -1
    item.max_id = -1


def propagate_deletions_once(map_of_squares, pair):
    """Second stage of remove_blocked_links: resolve one mutually-blocking pair
    (see mark_self_blocking_same_patch).

    Any item forcing *both* members of the pair is itself contradictory - it
    promised two things that can never both be chosen - so it gets clogged
    (clog_item). Once that's done, whichever member is left with no remaining
    .forced_by (every one of its forcers was one of the clogged common ones) is
    freed from the poisoned patch: it becomes its own fresh, independent patch
    (patch_id set to its own flattened index), propagated forward along its
    .forces to every item it reaches. A member that still has some other,
    non-common forcer left is untouched here - deferred to a later round.

    Returns True if this call clogged or reassigned anything, False otherwise -
    remove_blocked_links loops on this until a full round changes nothing.
    """
    rows, _ = map_of_squares.shape
    a_pos, b_pos = pair
    a_item = map_of_squares[a_pos]
    b_item = map_of_squares[b_pos]

    changed = False
    for pos in a_item.forced_by & b_item.forced_by:
        clog_item(map_of_squares, pos)
        changed = True

    for pos in (a_pos, b_pos):
        item = map_of_squares[pos]
        if not item.alert_chosen or item.forced_by:
            continue
        i, j = pos
        item.patch_id = i * rows + j
        to_visit = list(item.forces)
        visited = set()
        while to_visit:
            p = to_visit.pop()
            if p in visited:
                continue
            visited.add(p)
            map_of_squares[p].patch_id = item.patch_id
            to_visit.extend(map_of_squares[p].forces)
        changed = True

    return changed


def remove_blocked_links(map_of_squares, title=None, show=False):
    """Cleanup stage run once patch_id has converged (resolve_cycles_and_centrality):
    resolve every pair of alert_chosen items that diagonally block each other
    within the same patch (see mark_self_blocking_same_patch/
    propagate_deletions_once), re-scanning after each round since resolving one
    pair can change .forced_by elsewhere and surface a new contradiction. Stops
    once a full round resolves nothing - a pair that mark_self_blocking_same_patch
    still finds at that point (neither member's remaining forcers were all
    common) is left as-is, not an error.

    clog_item can leave behind a lone .blocked cell that completes some other
    2x2 block into a seat, so place_square_in_seat_closed runs next to resolve
    any of those. show=True displays a single before/after-style panel
    (display_closure_step) - but only if something actually changed, either
    from the pair resolution above or from a seat placed just now - a no-op
    call produces no display. Either way, the placements above leave
    .alert_chosen/.forces/etc. stale on the newly-chosen cells (see
    reset_alert_bookkeeping's docstring for why), so reset_alert_bookkeeping
    always runs last - with keep_patch_id_for_blocked=True, so a clog_item'd
    item's .patch_id (left alone by clog_item itself, since it's now
    permanently blocked) is still observable once this call returns, not just
    mid-call. The caller is expected to re-run find_alerts/link_patches itself
    if it needs fresh bookkeeping afterwards.
    """
    changed = False
    round_changed = True
    while round_changed:
        round_changed = False
        for pair in mark_self_blocking_same_patch(map_of_squares):
            if propagate_deletions_once(map_of_squares, pair):
                round_changed = True
        if round_changed:
            changed = True

    if place_square_in_seat_closed(map_of_squares):
        changed = True

    if show and changed:
        colormap = np.zeros((*map_of_squares.shape, 3))
        display_closure_step(map_of_squares, title or 'remove_blocked_links',
                              show_links=True, show_real=True, colormap=colormap)

def forced_closure(map_of_squares, position):
    """Every position transitively forced by position's own .forces (see
    SquareItem.forces), not including position itself: position's direct forces,
    plus whatever those force in turn, and so on, until every chain reaches a
    terminal (forces == set()) or loops back onto something already collected.

    This is the "actually commit to it" counterpart to link_patches/
    remove_blocked_links, which only ever *record* what choosing an item would
    oblige - nothing before this walks the recorded chain to say which positions
    that obligation actually reaches. Follows every entry in .forces, not just
    one: an item can force more than one other at once (see .forces'
    docstring), and only following a single arbitrary entry (as compute_blue_arrows
    does, for a single display arrow) would silently drop a real obligation. Makes no
    assumption that find_cycle_patches has already run - a forces chain can
    still loop back on itself at this stage - so each position is only ever
    visited once.
    

    A pure read - does not place anything itself. The caller places position,
    calls this, and places every position in the result too (see place_squares).
    """
    to_visit = list(map_of_squares[position].forces)
    forced = set()
    forced.add(position)
    while to_visit:
        pos = to_visit.pop()
        if pos in forced:
            continue
        forced.add(pos)
        to_visit.extend(map_of_squares[pos].forces)
    return forced


def reset_alert_bookkeeping(map_of_squares, keep_patch_id_for_blocked=False):
    """Clear every cell's .alert_blocked, .alert_chosen, .forces, .forced_by,
    .centrality, .patch_id, and .max_id back to their defaults (False,
    False, set(), set(), -1, -1, -1), map-wide, regardless of .state - so
    find_alerts/link_patches/resolve_cycles_and_centrality/remove_blocked_links
    can be re-run from a clean slate instead of layering new results on top of
    whatever an earlier round left behind.

    find_alerts and link_patches only ever add: both skip any cell whose .state
    isn't StateEnum.free, so once a cell is placed (e.g. as part of a
    forced_closure chase), whatever .alert_chosen/.alert_blocked/.forces/
    .forced_by it was carrying from an earlier round is never cleared - it just
    sits there, stale. That's silently wrong for display: colorize_with_alerts
    overlays alert_chosen/alert_blocked colour on top of the plain state colour,
    so a cell that is now genuinely chosen but still carries a stale
    alert_chosen=True renders as "promised, still free" (yellow) instead of
    actually placed (cyan) - and a stale .forces/.forced_by entry pointing at or
    from a no-longer-free cell is a dangling reference into a role that cell no
    longer plays. Needed whenever a round places more than one square at once
    (e.g. after chasing a forced_closure): the incremental single-step
    discipline the rest of closure.py assumes - one placement, then one
    find_alerts/link_patches pass - no longer applies once several cells change
    state in the same round, so the safe thing is to recompute every cell's
    alert bookkeeping from the current board state, not just the newly-placed
    ones.

    .centrality/.patch_id/.max_id need the same treatment once
    resolve_cycles_and_centrality/remove_blocked_links start running every round
    (see test_utils.place_and_chase): find_central_patch_items only ever assigns
    an item's .centrality once (`if item.centrality != -1: continue`), so a stale
    value surviving from an earlier round - where this same cell happened to be
    alert_chosen with different forces entirely - would silently block it from
    ever being reassigned in a later round.

    keep_patch_id_for_blocked=True skips clearing .patch_id, but only on cells
    already StateEnum.blocked (everything else above still resets as usual,
    on every cell regardless of state) - remove_blocked_links uses this so a
    clog_item'd item's .patch_id (deliberately left alone by clog_item itself
    - see its docstring) stays observable after remove_blocked_links returns
    too, not just mid-call. Deliberately scoped to blocked cells only: a
    still-free, still-alert_chosen cell (e.g. one on the surviving side of a
    2-cycle remove_blocked_links left untouched) needs a real -1 here, or a
    stale non--1 value left over from this same round confuses a later
    find_central_patch_items rebuild into thinking it's already been visited
    (`if q == -1: ...`), wrongly cutting its .forces as if retracing a cycle
    that was never actually retraced.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            item.alert_blocked = False
            item.alert_chosen = False
            item.forces = set()
            item.forced_by = set()
            item.centrality = -1
            if not (keep_patch_id_for_blocked and item.state == StateEnum.blocked):
                item.patch_id = -1
            item.max_id = -1


def check_tiling_invariant(map_of_squares):
    """Stage 3 of the closure pipeline: a 2x2 block of blocked items must never
    happen - the alert_blocked/alert_chosen/forces bookkeeping from stages 1 and 2
    exists specifically to prevent it. If it happens anyway, raise InvalidTilingError:
    that signals a bug upstream, not a recoverable case.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = (map_of_squares[i, j], map_of_squares[i, j + 1],
                       map_of_squares[i + 1, j], map_of_squares[i + 1, j + 1])
            if all(c.state == StateEnum.blocked for c in corners):
                raise InvalidTilingError(f"2x2 all-blocked block at ({i}, {j})")


def place_squares(map_of_squares, positions):
    """Set every (i, j) in positions to StateEnum.chosen on map_of_squares, in
    place, via set_square_chosen (so each one gets a chance to pair into a
    rectangle with an already-chosen direct neighbour). positions is expected
    to start out entirely free - e.g. a fresh map from build_map_of_squares -
    so no diagonal-neighbour pair among them can already be blocked as a side
    effect of some earlier placement.

    Blocks every diagonal neighbour of a placed square that is still free,
    matching the invariant map_of_squares_from_array enforces elsewhere
    (choosing an item blocks its four diagonal neighbours) - so a placed
    square's blocked neighbours show up on display_map_of_squares_3States too,
    not just the chosen square itself.
    """
    for i, j in positions:
        set_square_chosen(map_of_squares, (i, j))

    rows, cols = map_of_squares.shape
    for i, j in positions:
        for di, dj in DIAGONAL_OFFSETS:
            ni, nj = i + di, j + dj
            if (0 <= ni < rows and 0 <= nj < cols
                    and map_of_squares[ni, nj].state == StateEnum.free):
                map_of_squares[ni, nj].state = StateEnum.blocked


def place_square_in_seat(map_of_squares):
    """Scan every 2x2 block of adjacent map_of_squares cells (same scan as
    check_tiling_invariant) for a seat - three corners blocked, one free (see
    find_alerts's docstring) - and place a square at the free corner: the only
    alternative, letting that corner end up blocked too, is exactly the
    fully-blocked 2x2 check_tiling_invariant forbids.

    A direct state scan, independent of .alert_chosen bookkeeping - finds a
    seat wherever one currently exists on the board, not just where find_alerts
    already flagged one. Every seat found is placed in one batch (place_squares)
    rather than one at a time, so an earlier placement's diagonal-blocking
    side effect can't change a later seat's free corner out from under it
    mid-scan.

    Returns True if a seat was found (and placed), False otherwise -
    place_square_in_seat_closed loops on this until a call changes nothing.
    """
    rows, cols = map_of_squares.shape
    seats = set()
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
            states = [map_of_squares[p].state for p in corners]
            if states.count(StateEnum.blocked) == 3 and states.count(StateEnum.free) == 1:
                seats.add(corners[states.index(StateEnum.free)])

    if not seats:
        return False
    place_squares(map_of_squares, list(seats))
    return True


def place_square_in_seat_closed(map_of_squares):
    """Run place_square_in_seat to a fixed point: placing a square in one seat
    can block a diagonal neighbour that completes another 2x2 block into a
    fresh seat, so keep looping until a full call finds none left.

    Returns True if at least one seat was placed, False otherwise.
    """
    changed = False
    while place_square_in_seat(map_of_squares):
        changed = True
    return changed


def find_central_patch_items(map_of_squares, gen):
    """Assign centrality to alert_chosen items, one generation at a time.

    gen counts how many times this function has already been called for
    map_of_squares: call it with gen=0 first, then gen=1, 2, ... on each subsequent
    call. Requires find_alerts/link_patches to have already run (.alert_chosen/.forces
    must be set).

    The function runs over all alert_chosen items, checking every entry in .forces -
    not just an arbitrary one - since an item can force more than one other at once
    (see SquareItem.forces' docstring): checking only one entry would silently drop a
    real obligation, exactly the failure mode forced_closure's own docstring warns
    about. If gen=0 and the current item has a linked item with no .forces of its own,
    the linked item is a terminal item, and we set its centrality to 0, and its
    patch_id to its own flattened index (i * rows + j) - a fresh id for the patch that
    starts there.

    Each later call looks at every alert_chosen item that doesn't have a centrality
    yet: if any of its linked items has centrality gen-1, the current item's
    centrality becomes gen (an item is only ever assigned a centrality once - a
    second linked item resolving at some bigger gen in a later call can't clobber
    it). So centrality ends up measuring distance, in .forces hops, from a terminal
    item outward, one ring further with each generation.

    In that same step, patch_id propagates outward alongside centrality: the linked
    item (centrality gen-1, closer to the terminal) hands its patch_id to the current
    item (centrality gen, one step further out) -
      - if the current item has no patch_id yet (-1), it adopts the linked item's,
      - if it already has a *different* one (two patches have merged here), it keeps
        whichever of the two is larger - an arbitrary but consistent tie-break,
      - if it already has the *same* one, this item has already been reached by this
        patch once before - the only way that happens is by looping back around a
        cycle - so instead of reassigning, the current item's own .forces is cut
        (cleared to set()) to stop the patch from being retraced forever - retracting
        this item's own position from each old link's .forced_by first (see
        SquareItem.forced_by), so nothing downstream keeps listing it as a
        forcer it no longer is.

    A merge like that only fixes the patch_id of the item sitting at the merge point
    itself - everything closer to the terminal on the losing patch was already given
    the smaller id in some earlier call, before the merge was even discovered, and
    that's stale now. So on every call, every item that already has a patch_id
    additionally checks every one of its own .forces targets: if any of them now
    holds a *bigger* patch_id than the item itself has, the item adopts the largest
    one seen. Doing this every call lets a corrected id travel back down the losing
    patch, one hop per call, until the whole patch shares the same, single id -
    which is what makes patch_id double as a stable per-patch identifier in the end,
    without needing a separate pass to compute one: no additional "kernel calls"
    beyond the ones this function already needs for centrality.

    Reads for this correction step are all taken before any of its writes are
    applied (same snapshot-then-apply discipline as find_cycle_patches), so a
    patch_id correction can only move one hop per call, regardless of
    iteration order.

    Returns found: True if this call assigned a centrality, or propagated a
    patch_id correction, to at least one item, False otherwise - callers can
    loop, incrementing gen, until found comes back False to know everything
    reachable by .forces chains has fully converged.
    """
    rows, cols = map_of_squares.shape
    found = False
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.forces):
                continue

            if gen == 0:
                for target_pos in item.forces:
                    linked = map_of_squares[target_pos]
                    if not linked.forces:
                        linked.centrality = 0
                        li, lj = target_pos
                        linked.patch_id = li * rows + lj
                        found = True
                continue

            if item.centrality != -1:
                continue

            for target_pos in item.forces:
                linked = map_of_squares[target_pos]
                if linked.centrality != gen - 1:
                    continue
                item.centrality = gen
                found = True

                p = linked.patch_id
                q = item.patch_id
                if q == -1:
                    item.patch_id = p
                elif p != q:
                    item.patch_id = max(p, q)
                else:
                    for old_force in item.forces:
                        old_force_item = map_of_squares[old_force]
                        old_force_item.forced_by.discard((i, j))
                    item.forces = set()
                break

    corrections = {}
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.forces):
                continue
            for target_pos in item.forces:
                linked = map_of_squares[target_pos]

                if item.patch_id != -1 and linked.patch_id > item.patch_id:
                    best = corrections.get((i, j))
                    if best is None or linked.patch_id > best[1]:
                        corrections[i, j] = (item, linked.patch_id)

    for item, value in corrections.values():
        item.patch_id = value
    if corrections:
        found = True

    return found

def find_cycle_patches(map_of_squares, gen):
    """Ring leader election: find a parallel-safe break point for pure rings -
    patches with no terminal anywhere, so find_central_patch_items never assigns
    them a real centrality or patch_id.

    Every link is treated identically here - there's no reliance on visit order or
    which item happens to run first, only on each item's unique_id: its own
    flattened index (i * rows + j), the same convention find_central_patch_items
    uses to seed a terminal's patch_id.

    -- Known gap: fan-in candidates can still crowd each other out --
    More than one item can point at the same target via the (arbitrary) .forces
    entry this function reads - fan-in, the same shape link_patches's own .forces
    exists to support - so more than one candidate id
    can arrive at the same
    node in the same generation (test_do_closure_steps: (2, 2) receives a
    candidate from each of (2, 4), (3, 3), (4, 2), and (4, 4) at once, but only
    (3, 3) is actually on (2, 2)'s ring - the other three are tail branches that
    merely feed into it). max_id is a single scalar, so only one of several
    simultaneous candidates survives each generation - a list-based fix (one
    entry per still-live candidate) was tried and works, but was reverted in
    favour of a copy_map_reverse-based workaround (comparing cycle/centrality
    resolution on the map and a reversed copy side by side, to surface the
    "link out of a cycle" shape find_cycle_patches doesn't handle) to avoid
    the extra bookkeeping - that workaround has since been removed as unused,
    so this gap is currently unaddressed even as a stopgap. This scalar
    version is still subject to the crowding-out this describes; not fixed
    here.

    All reads below see map_of_squares exactly as it was at the start of this call
    - writes are collected and only applied once every edge has been evaluated, the
    same synchronisation a genuinely parallel pass over the ring would need before
    moving on to the next step. That's what makes "after m steps [m = ring size]
    there's only one [survivor]" a real guarantee: a candidate can move at most one
    edge per call, regardless of iteration order.

    For the current item A (a = A's unique_id, at position (i, j)) and its linked
    item B (B = map_of_squares[next(iter(A.forces))], an arbitrary but - for the
    duration of one call - stable pick since .forces isn't mutated mid-call):

    - If either A or B already has a real centrality, this pair isn't part of a
      pure ring (it's on a tree or a tadpole's tail) - max_id has no meaning there,
      so any max_id already sitting on either one is cleared back to -1.

    - gen == 0 (seed): every item sends its own identity one step along the ring -
      B.max_id = a. From here, that value either gets deleted somewhere along the
      way, or survives a full lap and lands back on the item it started from.

    - gen > 0, once A actually has a candidate sitting on it (A.max_id != -1),
      A compares that candidate against its own identity a:
        * a > A.max_id: A's own identity beats the incoming candidate - it dies,
          A.max_id is cleared.
        * a == A.max_id: the candidate is exactly A's own id, having survived a
          full lap of the ring back to where it started. Because every smaller
          candidate gets deleted somewhere along the way (a node with a bigger id
          is always encountered eventually), only the true maximum's own candidate
          can ever come back around - so this is the leader, confirmed. A.forces is
          cleared, opening the ring right here: A becomes an ordinary terminal
          (something points at it, it points at nothing), which is exactly what
          find_central_patch_items needs to seed centrality/patch_id from - once
          called again after the ring has been opened. Clearing A.forces also
          retracts A's own position from the .forced_by of each item A used to
          point at (see SquareItem.forced_by), so nothing downstream still lists A
          as a forcer it no longer is.
        * A.max_id > a: the candidate is still ahead of A - A does not beat it, so
          it moves on: A.max_id is cleared and B.max_id takes the value A had.
    """
    rows, cols = map_of_squares.shape
    resets = []
    seeds = []
    cuts = []
    propagations = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            A = map_of_squares[i, j]
            if not (A.alert_chosen and A.forces):
                continue
            B = map_of_squares[next(iter(A.forces))]

            if A.centrality != -1 or B.centrality != -1:
                if A.max_id != -1:
                    resets.append(A)
                if B.max_id != -1:
                    resets.append(B)
                continue

            a = i * rows + j

            if gen == 0:
                seeds.append((B, a))
                continue

            if A.max_id == -1:
                continue

            if a > A.max_id:
                resets.append(A)
            elif a == A.max_id:
                cuts.append((i, j, A))
            else:  # A.max_id > a
                resets.append(A)
                propagations.append((B, A.max_id))

    for target in resets:
        target.max_id = -1
    for target, value in seeds:
        target.max_id = value
    for i, j, target in cuts:
        for old_force in target.forces:
            old_force_item = map_of_squares[old_force]
            old_force_item.forced_by.discard((i, j))
        target.forces = set()
    for target, value in propagations:
        target.max_id = value


#    From here on the project is no longer aiming for a
#    parallel algorithm (for now).

def propagate_patch_id_from_roots(map_of_squares):
    """Stamp every item reachable, via .forces, from a "root" with the root's own
    patch_id.

    A root here is an alert_chosen item with .forces but no .forced_by: nothing
    forces it, so - unlike everything downstream of it - its own patch_id was never
    at the losing end of one of find_central_patch_items's merge decisions (see that
    function's docstring: patch_id only ever propagates from lower centrality
    outward to higher, so a losing branch at a merge point keeps its own smaller,
    stale id instead of adopting the merged one). A root's patch_id is therefore the
    most-settled id available for its whole component - this walks outward from
    every root along its own .forces links (same walk as forced_closure) and
    overwrites every reached item's patch_id with the root's.

    Not the last word on the matter - ensuring every item connected by links ends up
    sharing the same patch_id, in every graph shape, needs more than this (a
    component with more than one root, or none at all - e.g. a pure cycle before
    find_cycle_patches has opened it - isn't guaranteed to converge by this alone).
    Just an improvement over leaving a losing branch's stale id in place.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            root = map_of_squares[i, j]
            if not (root.alert_chosen and root.forces and not root.forced_by):
                continue
            if root.patch_id == -1:
                continue
            to_visit = list(root.forces)
            visited = set()
            while to_visit:
                pos = to_visit.pop()
                if pos in visited:
                    continue
                visited.add(pos)
                item = map_of_squares[pos]
                item.patch_id = root.patch_id
                to_visit.extend(item.forces)


def resolve_cycles_and_centrality(m, max_gens=20):
    """Run find_cycle_patches then find_central_patch_items to convergence on m,
    then propagate_patch_id_from_roots, in place.

    Fails for some graph types - parts of the graph will not be discovered.

    find_cycle_patches doesn't report whether it changed anything (unlike
    find_central_patch_items), so it's simply called a generous, fixed number of
    times (max_gens) rather than looped to a real convergence check - a
    stopgap, not a guarantee, matching this whole workaround's "lazy for now"
    scope.
    """
    for gen in range(max_gens):
        find_cycle_patches(m, gen)

    gen = 0
    while find_central_patch_items(m, gen):
        gen += 1

    propagate_patch_id_from_roots(m)


def redo_closure(m, title, show=False):
    find_alerts(m); link_patches(m)
    resolve_cycles_and_centrality(m)
    if show:
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap)
    remove_blocked_links(m)
    reset_alert_bookkeeping(m)
    find_alerts(m); link_patches(m)
    resolve_cycles_and_centrality(m)