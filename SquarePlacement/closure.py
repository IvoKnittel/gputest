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

def reset_alert_bookkeeping(map_of_squares):
    """Clear every cell's .alert_blocked, .alert_chosen, .forces, .forced_by,
    .centrality, .path_id, and .max_id back to their defaults (False,
    False, set(), set(), -1, set(), -1), map-wide, regardless of .state - so
    find_alerts/resolve_cycles_and_centrality
    can be re-run from a clean slate instead of layering new results on top of
    whatever an earlier round left behind.

    find_alerts only ever adds: it skips any cell whose .state
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
    find_alerts pass - no longer applies once several cells change
    state in the same round, so the safe thing is to recompute every cell's
    alert bookkeeping from the current board state, not just the newly-placed
    ones.

    .centrality/.path_id/.max_id need the same treatment once
    resolve_cycles_and_centrality start running every round
    (see test_utils.place_and_chase): find_central_patch_items only ever assigns
    an item's .centrality once (`if item.centrality != -1: continue`), so a stale
    value surviving from an earlier round - where this same cell happened to be
    alert_chosen with different forces entirely - would silently block it from
    ever being reassigned in a later round.
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
            if not (item.state == StateEnum.blocked):
                item.path_id = set()
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
    call. Requires find_alerts to have already run (.alert_chosen/.forces
    must be set).

    The function runs over all alert_chosen items, checking every entry in .forces.
    If gen=0 and the current item has a linked item with no .forces of its own,
    the linked item is a terminal item, and we set its centrality to 0, and its
    path_id to a fresh singleton set holding its own flattened index (i * rows + j) -
    a fresh id for the patch that starts there.

    Each later call looks at every alert_chosen item that doesn't have a centrality
    yet: if any of its linked items has centrality gen-1, the current item's
    centrality becomes gen (an item is only ever assigned a centrality once - a
    second linked item resolving at some bigger gen in a later call can't clobber
    it). So centrality ends up measuring how many .forces hops an item sits from the
    nearest terminal - one hop more with each generation.

    In that same step, item picks up its path_id alongside its centrality: item
    follows its own .forces - the causal, forward direction - to the linked item,
    whose centrality (gen-1) is lower, since centrality decreases going forward
    along .forces, down to 0 exactly at a terminal -
      - if the current item has no path_id yet (empty set), it adopts a copy of the
        linked item's,
      - if it already has a *different* one (two patches have merged here), an item
        can now belong to more than one patch at once, so the two id sets are
        unioned together rather than picking one via a tie-break,
      - if it already has the *same* one, this item has already been reached by this
        patch once before - the only way that happens is by looping back around a
        cycle - so instead of reassigning, the current item's own .forces is cut
        (cleared to set()) to stop the patch from being retraced forever - retracting
        this item's own position from each old link's .forced_by first (see
        SquareItem.forced_by), so nothing downstream keeps listing it as a
        forcer it no longer is.

    A merge like that only fixes the path_id of the item sitting at the merge point
    itself - everything closer to the terminal on the losing patch was already given
    the smaller id set in some earlier call, before the merge was even discovered, and
    that's stale now (missing whatever the merge just added). So on every call, every
    item that already has a path_id additionally checks every one of its own .forces
    targets: whatever id each target now holds that this item doesn't yet have is
    unioned in. Sets are only partially ordered - two force targets can each hold ids
    the other lacks without either being a superset of the other - so unioning in
    everything new from every target is the correct generalization of a single
    "biggest wins" tie-break: picking just one target's addition would silently drop
    whatever the others had. Doing this every call lets a correction move backward,
    one .forced_by hop at a time (from a target back to whatever forces it), until the
    whole patch shares the same, converged id set - which is what makes path_id
    double as a stable per-patch identifier in the end, without needing a separate
    pass to compute one: no additional "kernel calls" beyond the ones this function
    already needs for centrality.

    Reads for this correction step are all taken before any of its writes are
    applied (same snapshot-then-apply discipline as find_cycle_patches), so a
    path_id correction can only move one hop per call, regardless of
    iteration order.

    Returns found: True if this call assigned a centrality, or propagated a
    path_id correction, to at least one item, False otherwise - callers can
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
                        linked.path_id = {unique_id((li,lj), (rows, cols))}
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

                p = linked.path_id
                q = item.path_id
                if not q:
                    item.path_id = set(p)          # copy, don't alias
                elif p != q:
                    item.path_id = q | p           # union instead of max() tie-break
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
            if not (item.alert_chosen and item.forces and item.path_id):
                continue
            addition = set()
            for target_pos in item.forces:
                addition |= map_of_squares[target_pos].path_id - item.path_id
            if addition:
                corrections[i, j] = item.path_id | addition

    for pos, value in corrections.items():
        map_of_squares[pos].path_id = value
    if corrections:
        found = True

    return found

def find_cycle_patches(map_of_squares, gen):
    """Ring leader election: find a parallel-safe break point for pure rings -
    patches with no terminal anywhere, so find_central_patch_items never assigns
    them a real centrality or path_id.

    Every link is treated identically here - there's no reliance on visit order or
    which item happens to run first, only on each item's unique_id: its own
    flattened index (i * rows + j), the same convention find_central_patch_items
    uses to seed a terminal's path_id.

    -- Known gap: fan-in candidates can still crowd each other out --
    More than one item can point at the same target via the (arbitrary) .forces
    entry this function reads - fan-in, the same shape SquareItem.forces
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
          find_central_patch_items needs to seed centrality/path_id from - once
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

            a = unique_id((i,j), (rows, cols))

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


def propagate_path_id_from_entries(map_of_squares):
    """Union every self-seeded item's path_id forward, via .forces, into
    everything it reaches.

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
            if not item.path_id:
                continue
            
            if not unique_id((i,j), (rows, cols)) in item.path_id:
                continue

            entry=item
            to_visit = list(entry.forces)
            visited = set()
            while to_visit:
                pos = to_visit.pop()
                if pos in visited:
                    continue
                visited.add(pos)
                item = map_of_squares[pos]
                item.path_id = item.path_id | entry.path_id
                to_visit.extend(item.forces)

def unique_id(pos, size):
    """Flatten pos=(i, j) into a single id, unique per cell, given
    size=(rows, cols). i*M+j is only guaranteed collision-free if M >= cols
    (j never reaches M, so no row can overflow into the next one's range) -
    that held for the old i*rows+j formula only by accident, for every grid
    that happened to have rows >= cols. Multiply by whichever of rows/cols is
    the larger one: unchanged (i*rows+j) when rows >= cols, switching to
    i*cols+j only where the old formula would actually have collided (cols >
    rows - e.g. an 8x12 grid, where (0, 8) and (1, 0) both used to flatten to
    the same 8).
    """
    rows, cols = size
    if rows >= cols:
        return pos[0] * rows + pos[1]
    return pos[0] * cols + pos[1]

def assign_paths(map_of_squares):
    """Seed every entry, and every blocking-pair site, with its own path_id,
    then call propagate_path_id_from_entries to spread each seed forward
    along .forces.

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
    picked up an id from elsewhere in this same pass). This used to be
    remove_blocked_links's own first step; it's a seed exactly like an
    entry's, just keyed off .forced_by instead of "nothing forces it", so it
    belongs here alongside the rest of the seeding.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if item.forces and not item.forced_by:
                if len(item.forces) == 1:
                    target_pos = next(iter(item.forces))
                    target = map_of_squares[target_pos[0], target_pos[1]]
                    target.path_id.add(unique_id(target_pos, (rows, cols)))
                else:
                    item.path_id = {unique_id((i,j), (rows, cols))}

            if item.forced_by:
                for di, dj in DIAGONAL_OFFSETS:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < rows and 0 <= nj < cols):
                        continue
                    neighbour = map_of_squares[ni, nj]
                    if neighbour.state == StateEnum.free and neighbour.forced_by:
                        item.path_id.add(unique_id((i, j), (rows, cols)))
                        break

    propagate_path_id_from_entries(map_of_squares)


def resolve_cycles_and_centrality(m, max_gens=20):
    """Run find_cycle_patches then find_central_patch_items to convergence on m,
    then propagate_path_id_from_entries, in place.

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

    propagate_path_id_from_entries(m)


def forced_closure(map_of_squares, position):
    """position itself, plus every position transitively forced by its own
    .forces (see SquareItem.forces): position's direct forces, plus whatever
    those force in turn, and so on, until every chain reaches a terminal
    (forces == set()) or loops back onto something already collected.

    This is the "actually commit to it" counterpart to find_alerts/
    get_blocked_links/set_blocked_links, which only ever *record* what
    choosing an item would oblige - nothing before this walks the recorded
    chain to say which positions that obligation actually reaches. Follows
    every entry in .forces, not just
    one: an item can force more than one other at once (see .forces'
    docstring), and only following a single arbitrary entry would silently
    drop a real obligation. Makes no
    assumption that find_cycle_patches has already run - a forces chain can
    still loop back on itself at this stage - so each position is only ever
    visited once.

    A pure read - does not place anything itself. The caller places every
    position in the result, position included (see place_squares) - every
    call site does this as `place_squares(m, list(forced_closure(m, pos)))`,
    with no separate `+ [pos]`.
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

def get_blocked_links(m):
    """Return the set of path ids flagged as self-contradicting by any cell -
    ids, not positions. Run after assign_paths, not before: path_id has to
    already be real for this to mean anything.

    For every item A that has a path_id, build Q, the union of the path_id of
    every free diagonal neighbour A would block if A were ever chosen. S = Q
    & A.path_id is every id that both A and one of those soon-to-be-blocked
    neighbours share. A non-empty S is a direct contradiction: A belongs to a
    path (one of the ids in S) that choosing A would itself break, by
    blocking a fellow member of that same path. Every id any cell's own S
    contributes goes into the one set this function returns, regardless of
    how many different cells separately flag it - so the result names which
    paths are contradictory, not which cells witnessed the contradiction (see
    set_blocked_links for what happens with that set).

    A pure read - .path_id/.state are only ever looked at, never written, so
    unlike set_blocked_links this needs no snapshot-then-apply discipline of
    its own: nothing here can invalidate an earlier read.
    """
    rows, cols = m.shape
    p = set()
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
                if neighbour.state == StateEnum.free:
                    Q |= neighbour.path_id
            p |= (Q & A.path_id)
    return p


def set_blocked_links(m, p):
    """Set every cell whose own unique_id is in p (see get_blocked_links) to
    StateEnum.blocked_tmp - the origin cell of each self-contradicting path,
    not every cell that happens to carry one of its ids downstream - and
    clean up both consequences of p being self-contradicting.

    First, every cell's own path_id has every id in p removed, regardless of
    whether that cell itself ends up blocked_tmp: a self-contradicting path
    is broken, so nothing should still claim membership in it.

    Then, every newly-marked cell has its .forces/.forced_by cleared
    entirely, the same way clog_item used to: retracted from every other
    item's .forces/.forced_by first, so nothing downstream is left pointing
    at or from a role this cell no longer plays, then its own two sets
    emptied. A blocked_tmp cell ends up structurally indistinguishable from a
    genuinely StateEnum.blocked one - no dangling links - just a different
    .state value so display/debugging can still tell the two apart.

    Snapshot-then-apply for the marking step, same reason as
    get_blocked_links's own docstring: which cells qualify is decided by
    unique_id membership in p alone, not by anything that could change while
    this runs, so there's no ordering hazard there - but the positions to
    clear links from are still collected before any clearing starts, so
    clearing one cell's links can't affect which *other* cells get cleared.
    """
    rows, cols = m.shape

    for i in range(rows):
        for j in range(cols):
            m[i, j].path_id -= p

    to_clear = []
    for i in range(rows):
        for j in range(cols):
            if unique_id((i, j), (rows, cols)) in p:
                m[i, j].state = StateEnum.blocked_tmp
                to_clear.append((i, j))

    for pos in to_clear:
        item = m[pos]
        for target in item.forces:
            m[target].forced_by.discard(pos)
        for source in item.forced_by:
            m[source].forces.discard(pos)
        item.forces = set()
        item.forced_by = set()


def finalize_blocked_tmp(m):
    """Convert every StateEnum.blocked_tmp cell to StateEnum.blocked.

    set_blocked_links's own marking is provisional - "tmp" in the name is
    literal, it's a working flag for whatever inspects/reacts to a
    self-contradicting path, not a final verdict. Once nothing further is
    going to change based on that flag, this makes it permanent: a
    blocked_tmp cell already has empty .forces/.forced_by (set_blocked_links
    cleared those), so by this point the only thing distinguishing it from a
    cell that was always blocked is the .state value itself.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            if m[i, j].state == StateEnum.blocked_tmp:
                m[i, j].state = StateEnum.blocked


def do_closure(m, title, show=False, margin=None):
    """Run one full round of the closure pipeline, twice (see below for why
    twice), in place: find_alerts, assign_paths, get_blocked_links/
    set_blocked_links, finalize_blocked_tmp, place_square_in_seat_closed.

    margin (a representation.RealSpaceMargin, or None) is forwarded as-is to
    display_closure_step's own margin argument when show=True - see its
    docstring; ignored when show=False.

    The last two stages are the current standard, not just set_blocked_links:
    a cell get_blocked_links flags is a genuine, permanent impossibility (see
    test_get_and_set_blocked_links_marks_blocked_tmp's (5, 2) case - blocked
    on path_id grounds alone, with no diagonal-blocking neighbour to ever
    give it away locally), so blocked_tmp is finalized to a real
    StateEnum.blocked immediately, and place_square_in_seat_closed then fills
    in whatever seats that newly-permanent blocking completes - some of those
    same cells turn out to also be locally confirmed this way, but that's a
    bonus, not a requirement: the ones that aren't (like (5, 2)) are exactly
    the point of doing this at all.

    Runs the whole sequence twice: once with the optional display (so a
    caller sees the board after this round's own discoveries, before the
    next round's bookkeeping reset clears the alert/path state that produced
    them), then once more, silently, after reset_alert_bookkeeping - the same
    double-pass shape this function has always had, just with every stage of
    a single pass now included both times instead of only some of them.

    check_tiling_invariant runs once, at the very end, after both rounds -
    raising loudly (InvalidTilingError) rather than leaving an impossible
    2x2-all-blocked board to go unnoticed, which is what used to happen: the
    check existed but nothing ever called it. Confirmed this can actually
    happen: finalize_blocked_tmp/place_square_in_seat_closed can complete
    several seats in one batch (place_square_in_seat_closed's own scan-then-
    place-all-at-once discipline) without the per-placement re-scan that
    would otherwise catch a forming pinwheel - see the (2, 2)/(2, 3)/(3, 2)/
    (3, 3) case surfaced by test_margin_free_5x5realmap's very first round.
    """
    find_alerts(m)
    assign_paths(m)
    set_blocked_links(m, get_blocked_links(m))
    finalize_blocked_tmp(m)
    place_square_in_seat_closed(m)
    if show:
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap,
                              margin=margin)
    reset_alert_bookkeeping(m)
    find_alerts(m)
    assign_paths(m)
    set_blocked_links(m, get_blocked_links(m))
    finalize_blocked_tmp(m)
    place_square_in_seat_closed(m)
    check_tiling_invariant(m)