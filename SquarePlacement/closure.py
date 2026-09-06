import numpy as np

from map_of_squares import StateEnum, InvalidTilingError, set_square_chosen
from alert_graphs import (RING_OFFSETS,
                           DIAGONAL_OFFSETS,
                           iter_alert_thirds,
                           set_alert_blocked,
                           set_alert_chosen_set_links)
from representation import display_closure_step

def find_alerts_set_links(map_of_squares):
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

def find_secondary_links(map_of_squares):
    """
    A relay on top of find_alerts_set_links: for each alert_blocked item P,
    simulate each candidate placement a (a free diagonal neighbour of P - the
    thing that would actually cause P to become blocked) together with every
    corner B that blocking P already promises (P's own alert_chosen corners,
    from find_alerts_set_links), and look for seats that only exist once both
    of those - a's own diagonal-blocking footprint and B's - are combined.
    Link a directly to any such newly-created seat's corner.

    Inputs: reads .alert_blocked map-wide; for each alert_blocked P, .state of
    P's own ring (to recompute its corners) and of the diagonal neighbours of
    every position in the simulated chosen set (a and its corners); .state of
    every 2x2 block touching one of those.

    Outputs: writes .forces (on a), .forced_by and .alert_chosen (on each
    newly-linked corner); returns None.

    Scope: local - a fixed, bounded-radius simulation per (P, a) pair (P's
    ring, a's diagonal footprint, each corner's diagonal footprint, and the
    2x2 blocks touching any of that) - no traversal, no whole-board
    aggregate.

    -------------------------------------------------------------------------

    For an alert_blocked P, iter_alert_thirds(P's ring) gives B - the same
    corners find_alerts_set_links already found and linked from *every* free
    diagonal neighbour of P (it doesn't know which neighbour will actually
    trigger P's block, so it links them all). This function instead asks, one
    candidate a at a time: given that a specifically is what gets chosen -
    which blocks P, forcing every member of B to be chosen too, which in turn
    blocks *their* diagonal neighbours - does that combined, real chain of
    consequences complete a seat that isn't already one on the real board?
    If so, a is the one and only cause of it, so a (not B, not P) is what
    gets linked to the new corner.

    The simulation never needs an explicit local grid copy: only two kinds of
    cell ever change from the real board - the chosen set (a plus B) and
    whichever of their diagonal neighbours are currently free (blocked as a
    side effect) - so a small position->state override dict, consulted in
    place of the real .state, is enough. Only the 2x2 blocks touching an
    overridden position can possibly change seat status, so only those are
    rechecked; a block already a real seat before the override is skipped -
    that one belongs to place_square_in_seat_closed's direct scan already,
    not to this function.

    A found corner is skipped, not linked, if it's already .forced_by some
    member of B: find_alerts_set_links already links every member of B
    directly from a (a is one of P's free diagonal neighbours, and B is
    exactly what P's own alert_blocked/alert_chosen pass linked every such
    neighbour to), so a can already reach that corner in two hops via
    whichever b forces it - linking a to it directly would be a redundant
    edge, true but adding no reachability forced_closure didn't already have,
    and it only clutters the display. This one-hop-from-B check isn't a
    complete reachability test (a corner could be several hops past some b
    with nothing directly linking them), but it's cheap and catches the
    common case without turning this function's otherwise fixed-radius
    simulation into a graph walk.
    """
    rows, cols = map_of_squares.shape
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

            for di, dj in DIAGONAL_OFFSETS:
                a_pos = (i + di, j + dj)
                if not (0 <= a_pos[0] < rows and 0 <= a_pos[1] < cols):  # BR-005
                    continue
                a_item = map_of_squares[a_pos]
                if a_item.state != StateEnum.free:  # BR-006
                    continue

                chosen_hyp = {a_pos} | b_positions
                overrides = {pos: StateEnum.chosen for pos in chosen_hyp}
                for ci, cj in chosen_hyp:
                    for bdi, bdj in DIAGONAL_OFFSETS:
                        n_pos = (ci + bdi, cj + bdj)
                        if not (0 <= n_pos[0] < rows and 0 <= n_pos[1] < cols):  # BR-007
                            continue
                        if n_pos in overrides:  # BR-008
                            continue
                        if map_of_squares[n_pos].state == StateEnum.free:  # BR-009
                            overrides[n_pos] = StateEnum.blocked

                def eff_state(pos):
                    return overrides.get(pos, map_of_squares[pos].state)

                checked_blocks = set()
                for pi, pj in overrides:
                    for bi in (pi - 1, pi):
                        for bj in (pj - 1, pj):
                            if 0 <= bi < rows - 1 and 0 <= bj < cols - 1:  # BR-010
                                checked_blocks.add((bi, bj))

                for bi, bj in checked_blocks:
                    corners = [(bi, bj), (bi, bj + 1), (bi + 1, bj), (bi + 1, bj + 1)]

                    real_states = [map_of_squares[c].state for c in corners]
                    if (real_states.count(StateEnum.blocked) == 3
                            and real_states.count(StateEnum.free) == 1):  # BR-011
                        continue  # already a real seat - not newly created

                    hyp_states = [eff_state(c) for c in corners]
                    if (hyp_states.count(StateEnum.blocked) == 3
                            and hyp_states.count(StateEnum.free) == 1):  # BR-012
                        corner_pos = corners[hyp_states.index(StateEnum.free)]
                        corner_item = map_of_squares[corner_pos]
                        if corner_item.forced_by & b_positions:  # BR-013
                            continue  # already reachable from a via some b in B
                        a_item.forces.add(corner_pos)
                        corner_item.forced_by.add(a_pos)
                        corner_item.alert_chosen = True


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


def place_squares(map_of_squares, positions):
    """
    Set every (i, j) in positions to StateEnum.chosen on map_of_squares, in
    place, via set_square_chosen (so each one gets a chance to pair into a
    rectangle with an already-chosen direct neighbour). positions is expected
    to start out entirely free - e.g. a fresh map from build_map_of_squares -
    so no diagonal-neighbour pair among them can already be blocked as a side
    effect of some earlier placement.

    Inputs: reads .state of each diagonal neighbour of each position (to
    check it's still free before blocking it).

    Outputs: writes .state (free->chosen, plus .rectangle pairing via
    set_square_chosen) on positions, and .state (free->blocked) on their
    diagonal neighbours; returns None.

    Scope: local - each position's effect is confined to its own
    fixed 1-hop neighbourhood, independent of every other position passed in.

    -------------------------------------------------------------------------

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
                    and map_of_squares[ni, nj].state == StateEnum.free):  # BR-015
                map_of_squares[ni, nj].state = StateEnum.blocked


def place_square_in_seat(map_of_squares):
    """
    Scan every 2x2 block of adjacent map_of_squares cells (same scan as
    check_tiling_invariant) for a seat - three corners blocked, one free (see
    find_alerts_set_links's docstring) - and place a square at the free corner:
    the only alternative, letting that corner end up blocked too, is exactly the
    fully-blocked 2x2 check_tiling_invariant forbids.

    Inputs: reads .state of every 2x2 block of adjacent cells.

    Outputs: writes .state (free->chosen, plus diagonal blocking via
    place_squares) at every seat found; returns a bool.

    Scope: local - each block's own check reads only its own 4 cells; every
    seat found across the whole scan is collected before any of them is
    placed (see below for why), but that collection is still a plain
    local-per-block result, not a graph walk or a global-identity aggregate
    the way get_blocked_links's return value is.

    -------------------------------------------------------------------------

    A direct state scan, independent of .alert_chosen bookkeeping - finds a
    seat wherever one currently exists on the board, not just where
    find_alerts_set_links already flagged one. Every seat found is placed in one batch (place_squares)
    rather than one at a time, so an earlier placement's diagonal-blocking
    side effect can't change a later seat's free corner out from under it
    mid-scan.

    -- Known gap: two seats in the same scan can be mutually diagonal --
    Confirmed by direct repro (build a margin-blocked board, run real square
    placement through it - see Quality/test_image_to_squares.py's
    test_square_placement_random_order_supersuperlattice): two 2x2 blocks found
    as seats in the *same* scan can themselves be diagonal neighbours of each
    other. When that happens there is no locally-correct resolution:

    - Choosing both (today's behaviour) violates the no-diagonal-chosen-pair
      invariant real_space_map enforces - check_tiling_invariant doesn't catch
      this, since it only checks for a fully-blocked 2x2.
    - Blocking either one instead immediately turns THAT one's own
      already-3-blocked 2x2 fully blocked, tripping check_tiling_invariant
      directly.
    - Deferring one and protecting it from the other's diagonal-blocking step
      only postpones the same conflict: nothing else
      in this pipeline stops a later round from choosing a free cell whose
      diagonal neighbour is already chosen, so the deferred seat gets chosen on
      its own a few rounds later and the exact same violation reappears.

    Every local fix attempted here changes *which* invariant breaks, never
    prevents both. That means this state - two independently-forced seats that
    are mutually diagonal - shouldn't be reachable in the first place: this
    function's own docstring already flags it as "independent of .alert_chosen
    bookkeeping", i.e. it bypasses the whole find_alerts_set_links/assign_paths/
    get_blocked_links/dissolve_blocked_paths promise-and-contradiction system
    (see get_blocked_links's docstring) that exists specifically to catch a
    self-contradicting pair *before* it calcifies into two simultaneously-forced
    cells. The real fix belongs upstream of this function, not inside it - not
    attempted here.

    test_sudden_appearance.py used to be concrete evidence of exactly that
    bypass - test_seat_from_two_alert_blocked and test_frozen_area each
    placed one square and showed a *different*, distant cell end up
    StateEnum.chosen with an empty .forced_by throughout, chosen purely by
    this function's own scan, never recorded by find_alerts_set_links/
    assign_paths at all. find_secondary_links (closure.py) now catches both
    of those two specific cases - see its own docstring for how - so those
    two tests now assert a real .forced_by instead. The general gap above is
    still open, though: find_secondary_links only catches a seat that forms
    by combining one alert_blocked item's own diagonal-blocking footprint
    with a corner it already promises - not every way a seat can form
    without either corner being locally visible beforehand. get_blocked_links
    only ever checks path_id membership, so a cell chosen this function's own
    way - no path_id, no .forced_by - is still invisible to it whenever
    find_secondary_links doesn't happen to cover the shape. Two such choices
    happening to be diagonal neighbours of each other is exactly the gap
    above, and it doesn't currently have a live repro in the suite.

    Returns True if a seat was found (and placed), False otherwise -
    place_square_in_seat_closed loops on this until a call changes nothing.
    """
    rows, cols = map_of_squares.shape
    seats = set()
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
            states = [map_of_squares[p].state for p in corners]
            if states.count(StateEnum.blocked) == 3 and states.count(StateEnum.free) == 1:  # BR-016
                seats.add(corners[states.index(StateEnum.free)])

    if not seats:  # BR-017
        return False
    place_squares(map_of_squares, list(seats))
    return True


def place_square_in_seat_closed(map_of_squares):
    """
    Run place_square_in_seat to a fixed point: placing a square in one seat
    can block a diagonal neighbour that completes another 2x2 block into a
    fresh seat, so keep looping until a full call finds none left.

    Inputs: none of its own - delegates entirely to place_square_in_seat.

    Outputs: same as place_square_in_seat, applied repeatedly; returns a bool.

    Scope: local-global - each place_square_in_seat call is a local scan, but
    looping it to a fixed point is what lets one placement's diagonal-blocking
    side effect reach a seat anywhere else on the board.

    -------------------------------------------------------------------------

    Returns True if at least one seat was placed, False otherwise.
    """
    changed = False
    while place_square_in_seat(map_of_squares):
        changed = True
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

def assign_paths(map_of_squares):
    """
    Seed every entry, and every blocking-pair site, with its own path_id,
    then call propagate_path_id_from_entries to spread each seed forward
    along .forces.

    Inputs: reads .forces, .forced_by of every cell, plus .state and
    .forced_by of each cell's diagonal neighbours (for the self-blocking-pair
    seed).

    Outputs: writes .path_id (seeds), then calls
    propagate_path_id_from_entries (a global write - see its own header);
    returns None.

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


def forced_closure(map_of_squares, position):
    """
    position itself, plus every position transitively forced by its own
    .forces (see SquareItem.forces): position's direct forces, plus whatever
    those force in turn, and so on, until every chain reaches a terminal
    (forces == set()) or loops back onto something already collected.

    Inputs: reads .forces of position and of every cell transitively
    reached via .forces.

    Outputs: returns a set of positions; writes nothing to the map (a pure
    read).

    Scope: global - an explicit BFS across .forces, unbounded in reach, the
    same shape as propagate_path_id_from_entries.

    -------------------------------------------------------------------------

    This is the "actually commit to it" counterpart to find_alerts_set_links/
    get_blocked_links/dissolve_blocked_paths, which only ever *record* what
    choosing an item would oblige - nothing before this walks the recorded
    chain to say which positions that obligation actually reaches. Follows
    every entry in .forces, not just
    one: an item can force more than one other at once (see .forces'
    docstring), and only following a single arbitrary entry would silently
    drop a real obligation. Makes no
    assumption that a pure .forces cycle has been broken anywhere else - a
    forces chain can still loop back on itself - so each position is only
    ever visited once.

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
    """
    Return the set of path ids flagged as self-contradicting by any cell -
    ids, not positions. Run after assign_paths, not before: path_id has to
    already be real for this to mean anything.

    Inputs: reads .path_id map-wide (plus positions, to look up each cell's
    diagonal neighbours - no other field).

    Outputs: returns a set of path ids; writes nothing to the map.

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
    return p


def dissolve_blocked_paths(m, p):
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
    whose own unique_id is a member; returns None.

    Scope: local per id - unique_id is injective, so each id in p names
    exactly one cell; this is a bounded, single-write-per-id pass, not a
    board-wide aggregate.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            if unique_id((i, j), (rows, cols)) in p:
                m[i, j].state = StateEnum.blocked


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
    looping place_square_in_seat.

    Returns True if at least one cell was newly blocked this way, False
    otherwise.
    """
    changed = False
    while propagate_blocked_tmp(m):
        changed = True
    return changed


def do_closure(m, title, show=False, margin=None, roi_margin=0):
    """
    Run one full round of the closure pipeline, twice (see below for why
    twice), in place: find_alerts_set_links, assign_paths, get_blocked_links/
    dissolve_blocked_paths, place_square_in_seat_closed.

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
    get_blocked_links, dissolve_blocked_paths, place_square_in_seat,
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

    margin (a representation.RealSpaceMargin, or None) and roi_margin are
    forwarded as-is to display_closure_step's own margin/roi_margin arguments
    when show=True - see their docstrings; ignored when show=False.

    show=True's display (after the first pass, before the bookkeeping reset -
    see below) also raises InvalidTilingError if it finds two chosen squares
    that are diagonal neighbours - display_closure_step's show_real=True panel
    reports that via its own return value (real_space_map does not raise
    it directly, see its docstring), and this is the one place that turns it
    back into a raise, matching check_tiling_invariant's already-loud handling
    of the other kind of invalid board (a fully-blocked 2x2). show=False skips
    this check entirely, the same way it skips the display itself - a
    diagonal-chosen conflict can still be present on a show=False run, just
    undetected by do_closure itself either way.

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
    """
    find_alerts_set_links(m)
    find_secondary_links(m)
    assign_paths(m)
    dissolve_blocked_paths(m, get_blocked_links(m))
    place_square_in_seat_closed(m)
    if show:
        colormap = np.zeros((*m.shape, 3))
        error = display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap,
                                      margin=margin, roi_margin=roi_margin)
        if error:
            raise InvalidTilingError(
                f"{title}: real_space_map found a diagonal-chosen conflict - "
                f"see the map_of_squares panel just shown for which cells")
    clear_all_but_state(m)
    find_alerts_set_links(m)
    find_secondary_links(m)
    assign_paths(m)
    dissolve_blocked_paths(m, get_blocked_links(m))
    place_square_in_seat_closed(m)
    check_tiling_invariant(m)