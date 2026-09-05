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
            if item.state != StateEnum.free:
                continue
            ring = [map_of_squares[i + di, j + dj] for di, dj in RING_OFFSETS]
            set_alert_blocked(item, ring)
            if item.alert_blocked:
                set_alert_chosen_set_links(i, j, ring)

def find_secondary_links(map_of_squares):
    """
    A relay on top of find_alerts_set_links: an item C that just became
    alert_chosen is certain to be chosen - and choosing it really blocks its own
    four diagonal neighbours (see place_squares). So each of C's still-free
    diagonal neighbours D is certain to become blocked too - not "if", the way
    find_alerts_set_links's own alert_blocked is, but for sure.

    Inputs: reads .alert_chosen map-wide, and .state of each alert_chosen
    cell's diagonal neighbours plus that neighbour's own 8-neighbour ring.

    Outputs: writes .alert_blocked (on the diagonal neighbour), .alert_chosen/
    .forced_by (on the corner found), and .forces (on the original alert_chosen
    cell); returns None.

    Scope: local - a fixed 2-hop read radius per cell, no
    traversal and no whole-board aggregate returned.

    -------------------------------------------------------------------------

    Run the exact same
    seat check find_alerts_set_links runs for a hypothetically-blocked item
    (iter_alert_thirds on D's own ring) to see whether D's now-certain blocking
    completes a seat elsewhere - a fresh alert_chosen corner find_alerts_set_links's
    single real-state pass, run before any of this was known, couldn't have seen.

    find_alerts_set_links links every one of an alert_blocked item's free diagonal
    neighbours to the seat's corner, because it doesn't know which of them will end
    up triggering the block. Here the trigger is already known exactly - C itself -
    so it's C, not each of D's own diagonal neighbours, that gets linked to the new
    corner: C.forces gains it, and it gets C added to its own .forced_by. A corner
    landing back on C's own position (D is diagonal to C, so C sits in D's ring
    too) is skipped - C is presently alert_chosen, hence still StateEnum.free, so
    iter_alert_thirds can otherwise mistake it for a genuine free completer of its
    own seat.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            c_item = map_of_squares[i, j]
            if not c_item.alert_chosen:
                continue

            for di, dj in DIAGONAL_OFFSETS:
                di_pos, dj_pos = i + di, j + dj
                if not (1 <= di_pos < rows - 1 and 1 <= dj_pos < cols - 1):
                    continue

                d_item = map_of_squares[di_pos, dj_pos]
                if d_item.state != StateEnum.free:
                    continue

                d_item.alert_blocked = True
                ring = [map_of_squares[di_pos + rdi, dj_pos + rdj] for rdi, rdj in RING_OFFSETS]
                for third_idx in iter_alert_thirds(ring):
                    ti = di_pos + RING_OFFSETS[third_idx][0]
                    tj = dj_pos + RING_OFFSETS[third_idx][1]
                    if (ti, tj) == (i, j):
                        continue
                    corner = map_of_squares[ti, tj]
                    corner.alert_chosen = True
                    c_item.forces.add((ti, tj))
                    corner.forced_by.add((i, j))


def clear_all_but_state(map_of_squares):
    """
    Clear every cell's .blocked_tmp, .alert_blocked, .alert_chosen, .forces,
    .forced_by, .centrality, .path_id, and .max_id back to their defaults
    (False, False, False, set(), set(), -1, set(), -1), map-wide,
    unconditionally - .state is the only field that starts a round and
    survives it; everything else is derived fresh from .state each round, so
    nothing else should carry over. This is what lets find_alerts_set_links/
    assign_paths run from a clean slate instead of layering new results on
    top of whatever an earlier round left behind, and it's also the only
    place .blocked_tmp ever gets cleared: set_blocked_links's own .state
    write is already permanent the moment it happens (see its docstring), so
    clearing the flag here is bookkeeping cleanup, not a second finalization
    step - nothing needs to convert .state from anything to anything else.

    Inputs: none - every field is cleared to a fixed default, regardless of
    .state or anything else already on the cell.

    Outputs: writes .blocked_tmp, .alert_blocked, .alert_chosen, .forces,
    .forced_by, .centrality, .path_id, and .max_id on every cell,
    unconditionally; returns None.

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

    .centrality and .max_id are both currently dead: they belonged to a
    separate centrality/path_id mechanism (find_central_patch_items,
    find_cycle_patches) that has since been removed entirely, and nothing
    else in the codebase writes or reads either field. Cleared here anyway,
    same as everything else: a reset should not leave any field depending on
    what happened to be true before it ran.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            item.blocked_tmp = False
            item.alert_blocked = False
            item.alert_chosen = False
            item.forces = set()
            item.forced_by = set()
            item.centrality = -1
            item.path_id = set()
            item.max_id = -1


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
            if all(c.state == StateEnum.blocked for c in corners):
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
                    and map_of_squares[ni, nj].state == StateEnum.free):
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
    get_blocked_links/set_blocked_links promise-and-contradiction system (see
    get_blocked_links's docstring) that exists specifically to catch a
    self-contradicting pair *before* it calcifies into two simultaneously-forced
    cells. The real fix belongs upstream of this function, not inside it - not
    attempted here.

    Concrete evidence of exactly that bypass, already in the suite:
    test_sudden_appearance.test_seat_from_two_alert_blocked places one
    square and shows a *different*, distant cell end up StateEnum.chosen with
    an empty .forced_by throughout - chosen purely by this function's own scan,
    never recorded by find_alerts_set_links/assign_paths at all. get_blocked_links only
    ever checks path_id membership, so a cell chosen this way - no path_id, no
    .forced_by - is invisible to it. Two such choices happening to be diagonal
    neighbours of each other is exactly the gap above.

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
    if rows >= cols:
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
    get_blocked_links/set_blocked_links, which only ever *record* what
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
    witnessed the contradiction (see set_blocked_links for what happens with
    that set).

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
    unlike set_blocked_links this needs no snapshot-then-apply discipline of
    its own: nothing here can invalidate an earlier read.

    -- Flagged for rewrite: global id set + a second full-grid scan to match it --
    This function's whole output is a set of ids with no positional
    information; set_blocked_links then has to re-scan every cell
    (`unique_id((i, j), ...) in p`) just to find which cells those ids
    actually belong to. That round trip through unique_id/a global set
    works, but it's not in place: propagating
    the contradiction directly along each cell's own .forces/.forced_by links
    (the same links assign_paths/propagate_path_id_from_entries already walk)
    instead of round-tripping through a global id set would let get_blocked_links
    mark the origin cells itself, with no second grid-wide scan needed. Not
    done yet: noted here as a target, not attempted.

    Also flagged for rewrite: this global id vector, as get_blocked_links's
    output and set_blocked_links's input, breaks the GPU/tile computing model
    do_closure's own "Flagged for rewrite" note aims for.
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
                if neighbour.path_id:
                    Q |= neighbour.path_id
            p |= (Q & A.path_id)
    return p


def set_blocked_links(m, p):
    """
    Set every cell whose own unique_id is in p (see get_blocked_links) to
    StateEnum.blocked, with its .blocked_tmp flag raised to mark it as the
    origin of a self-contradicting path - not every cell that happens to
    carry one of its ids downstream - and clean up both consequences of p
    being self-contradicting.

    Inputs: reads p (a global set of ids, from get_blocked_links) against
    every cell's position, plus .path_id (to subtract p from), and .forces/
    .forced_by of every flagged cell and of everything they point to or are
    pointed from.

    Outputs: writes .state (free->blocked on flagged cells) and .blocked_tmp
    (False->True on the same cells) - a state change to chosen/blocked goes
    with deleting that cell's own .path_id and .forces/.forced_by, so that
    part needs no separate mention. Distinct from that: subtracts p from
    every cell's .path_id map-wide, including cells that never change state
    at all - that part is a real, separate effect this function has. Returns
    None.

    Scope: global - p is itself a whole-board-scope value with no positional
    information, so resolving which cells it names requires scanning every
    cell against it (see get_blocked_links's own "Flagged for rewrite" note).

    -------------------------------------------------------------------------

    First, every cell's own path_id has every id in p removed, regardless of
    whether that cell itself ends up flagged: a self-contradicting path
    is broken, so nothing should still claim membership in it.

    Then, every newly-flagged cell has its .forces/.forced_by cleared
    entirely: retracted from every other
    item's .forces/.forced_by first, so nothing downstream is left pointing
    at or from a role this cell no longer plays, then its own two sets
    emptied. Once that's done, a .blocked_tmp cell is structurally
    indistinguishable from a cell that was always blocked - no dangling
    links, real StateEnum.blocked .state already in place - .blocked_tmp is
    purely an extra marker so display/debugging can still tell the two apart.

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
                m[i, j].blocked_tmp = True
                m[i, j].state = StateEnum.blocked
                to_clear.append((i, j))

    for pos in to_clear:
        item = m[pos]
        for target in item.forces:
            m[target].forced_by.discard(pos)
        for source in item.forced_by:
            m[source].forces.discard(pos)
        item.forces = set()
        item.forced_by = set()


def do_closure(m, title, show=False, margin=None):
    """
    Run one full round of the closure pipeline, twice (see below for why
    twice), in place: find_alerts_set_links, assign_paths, get_blocked_links/
    set_blocked_links, place_square_in_seat_closed.

    Inputs: none of its own - delegates entirely to the stages it calls, in
    sequence.

    Outputs: writes essentially every SquareItem field via those stages;
    returns None, or raises InvalidTilingError.

    Scope: global - includes several explicitly global stages of its own
    (assign_paths, get_blocked_links/set_blocked_links), so the pipeline as a
    whole is global regardless of how local its individual stages are.

    -------------------------------------------------------------------------

    -- Flagged for rewrite: cell-by-cell Python loops, not GPU-style tiles --
    Every stage this orchestrates (find_alerts_set_links, assign_paths,
    get_blocked_links, set_blocked_links, place_square_in_seat,
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

    margin (a representation.RealSpaceMargin, or None) is forwarded as-is to
    display_closure_step's own margin argument when show=True - see its
    docstring; ignored when show=False.

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

    place_square_in_seat_closed follows set_blocked_links because a cell
    get_blocked_links flags is a genuine, permanent impossibility (see
    test_get_and_set_blocked_links_marks_blocked_tmp's (5, 2) case - blocked
    on path_id grounds alone, with no diagonal-blocking neighbour to ever
    give it away locally) - set_blocked_links already writes the real
    StateEnum.blocked immediately (.blocked_tmp is just a marker alongside
    it, not a separate pending state), so place_square_in_seat_closed can
    fill in whatever seats that newly-permanent blocking completes right
    away, no finalization step needed in between. Some of those same cells
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
    would otherwise catch a forming pinwheel - see the (2, 2)/(2, 3)/(3, 2)/
    (3, 3) case surfaced by test_margin_free_5x5realmap's very first round.
    """
    find_alerts_set_links(m)
    assign_paths(m)
    set_blocked_links(m, get_blocked_links(m))
    place_square_in_seat_closed(m)
    if show:
        colormap = np.zeros((*m.shape, 3))
        error = display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap,
                                      margin=margin)
        if error:
            raise InvalidTilingError(
                f"{title}: real_space_map found a diagonal-chosen conflict - "
                f"see the map_of_squares panel just shown for which cells")
    clear_all_but_state(m)
    find_alerts_set_links(m)
    assign_paths(m)
    set_blocked_links(m, get_blocked_links(m))
    place_square_in_seat_closed(m)
    check_tiling_invariant(m)