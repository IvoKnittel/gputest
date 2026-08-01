import dataclasses
import random

import numpy as np

from map_of_squares import StateEnum, InvalidTilingError, CoreItem
from alert_graphs import (RING_OFFSETS,
                           DIAGONAL_OFFSETS,
                           set_alert_blocked,
                           set_alert_chosen,
                           resolve_chosen_link,
                           mark_patch_conflicts as mark_item_patch_conflicts)


def fill_isolated_free_cells_once(map_of_squares):
    """set state of isolated free items:
    isolated item : none of the 4 direct (orthogonal) neighbours are all none are free.
    set item chosen except if two neighbors are chosen.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            neighbours = (map_of_squares[i - 1, j], map_of_squares[i + 1, j],
                          map_of_squares[i, j - 1], map_of_squares[i, j + 1])
            if all(n.state != StateEnum.free for n in neighbours):
                found_chosen_once=False
                item.state = StateEnum.chosen
                for n in neighbours:
                    if n.state == StateEnum.chosen:
                        if found_chosen_once:
                           item.state = StateEnum.blocked
                           for forced_item_idx in item.forces:
                               map_of_squares[forced_item_idx].forced_by.discard((i, j))
                           item.forces = set()
                        found_chosen_once=True

def fill_isolated_free_cells(map_of_squares):
    """set state of isolated free items:
    isolated item : none of the 4 direct (orthogonal) neighbours are all none are free.
    set item chosen except if two neighbors are chosen.

    function must be re-run because condition "two neighbors are chosen" may apply on other items.
    """
    fill_isolated_free_cells_once(map_of_squares)
    fill_isolated_free_cells_once(map_of_squares)


def find_alerts(map_of_squares):
    """find_patches stage 2 (set_alert_blocked, set_alert_chosen, iter_alert_thirds):
    a seat (team term - see docs/rose_cascades_and_holes/README.md - for what this
    function's own name calls an "alert" as a noun) is a 2x2 block with three items
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
    """find_patches stage 3 - was: adopt, for every free item, whatever
    resolve_chosen_link's diagonal-neighbour relay found, replacing that item's
    forces with it. resolve_chosen_link is now a guaranteed no-op (see its
    docstring: set_alert_chosen already links every free diagonal neighbour of
    an alert_blocked centre directly, so there is nothing left for a relay stage
    to find - the relay this function used to perform is not merely redundant,
    it actively overwrote correct stage-2 forces with wrong ones wherever it
    used to fire). So this function no longer changes anything either; it is
    kept, rather than removed, purely so every existing caller (find_patches,
    every test's pipeline) keeps working unchanged.
    """
    rows, cols = map_of_squares.shape
    updates = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            new_forces = resolve_chosen_link(i, j, map_of_squares)
            if new_forces:
                updates.append((i, j, item, new_forces))
    for i, j, item, new_forces in updates:
        item.alert_chosen = True
        for old_target in item.forces:
            old_target_item = map_of_squares[old_target]
            old_target_item.forced_by.discard((i, j))
        item.forces = new_forces
        for target_pos in new_forces:
            map_of_squares[target_pos].forced_by.add((i, j))


def remove_blocked_links(map_of_squares):
    """Cleanup stage after link_patches: an item's own .forces can end up naming a
    target that is actually guaranteed to end up blocked, not chosen - something
    link_patches itself doesn't catch, since it only ever looks at an item's own
    diagonal neighbours, never at what its own .forced_by already commit it to.

    Concretely (see test_do_closure_steps): (6, 4).forces == {(6, 3)}, asserting
    that choosing (6, 4) also requires choosing (6, 3). But (7, 2) is in
    (6, 4).forced_by - (7, 2).forces contains (6, 4), so (7, 2) is itself
    guaranteed to be chosen alongside (6, 4) (that's exactly what a forced_by
    entry means). Choosing (7, 2) blocks all 4 of its own diagonal neighbours -
    and (6, 3) is one of them (DIAGONAL_OFFSETS from (7, 2) reach (6, 1), (6, 3),
    (8, 1), (8, 3)). So (6, 3) is doomed to end up blocked by (7, 2) at the same
    time (7, 2) forces (6, 4) - making (6, 4)'s own link to (6, 3)
    self-contradictory: it names a target that can never actually be chosen.

    For every item, and every item Y in its .forced_by (an item guaranteed
    chosen, since Y is what forced this one), any of Y's diagonal neighbours that
    also appears in this item's own .forces is removed - Y being chosen is exactly
    what dooms that neighbour to end up blocked instead of chosen. Removing it
    also retracts this item's own position from that target's .forced_by, so
    the target no longer lists a forcer that no longer actually forces it (see
    SquareItem.forced_by) - the same retraction link_patches does when it
    replaces an item's forces.

    A single pass, not a fixed point: removing one invalid link here could, in
    principle, itself change what some other item's .forced_by commit it to,
    invalidating a further link elsewhere. Not chased down further for now - the
    simplest version, on the assumption it's already enough.
    """
    rows, cols = map_of_squares.shape
    removals = []
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.forces:
                continue
            for yi, yj in item.forced_by:
                for di, dj in DIAGONAL_OFFSETS:
                    blocked_pos = (yi + di, yj + dj)
                    if blocked_pos in item.forces:
                        removals.append((i, j, item, blocked_pos))

    for i, j, item, target in removals:
        if target in item.forces:
            item.forces.remove(target)
            target_item = map_of_squares[target]
            target_item.forced_by.discard((i, j))


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


def reset_alert_bookkeeping(map_of_squares):
    """Clear every cell's .alert_blocked, .alert_chosen, .forces, and .forced_by
    back to their defaults (False, False, set(), set()), map-wide, regardless of
    .state - so find_alerts/link_patches/remove_blocked_links can be re-run from
    a clean slate instead of layering new flags on top of whatever an earlier
    round left behind.

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
    """
    rows, cols = map_of_squares.shape
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            item.alert_blocked = False
            item.alert_chosen = False
            item.forces = set()
            item.forced_by = set()


def copy_map_reverse(map_of_squares):
    """Build a copy of map_of_squares, identical in every field except .forces,
    which point the other way: for every edge source -> target in the original,
    the copy has target -> source instead.

    -- Why this exists: the "link out of a cycle" case find_cycle_patches doesn't
    handle --
    find_cycle_patches's ring-leader election assumes every node it looks at is
    either on the ring itself, or a tail feeding into it (see the "why max_ids is
    a list" note there) - candidates from a tail always eventually lose to a
    bigger id and die out, never disturbing the ring's own leader election. But a
    link running the other way - out of a ring, to some node beyond it, rather
    than into it - isn't one of those two shapes, and isn't handled: nothing
    currently walks a ring's members looking for one of *their* diagonal
    neighbours needing a link back out again. Reversing every link turns exactly
    that case into an ordinary tail (now feeding out of, rather than into, the
    reversed ring) - at the cost of turning genuine tails into the same
    unhandled shape instead. Running the same cycle/centrality resolution on both
    the original and the reversed copy, and comparing the two side by side (see
    test_do_closure_steps_reverse_check), is a stopgap for surfacing which shape
    is actually present - not a real fix for handling both at once.

    Every other field (.state, .alert_chosen, .alert_blocked, .quality,
    .centrality, .patch_id, .max_id) is copied as-is; .conflicts is copied as a
    new set (not the same set object) so mutating one copy's .conflicts - e.g.
    during find_central_patch_items - can never leak into the other's. .forces
    and .forced_by are rebuilt from scratch, reversed relative to the
    original (see SquareItem.forced_by) - each of the copy's own .forces
    entries also appends the matching .forced_by entry, the same as every
    other place .forces gets created.
    """
    rows, cols = map_of_squares.shape
    reversed_map = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            reversed_map[i, j] = dataclasses.replace(
                item, forces=set(), forced_by=set(), conflicts=set(item.conflicts))

    for i in range(rows):
        for j in range(cols):
            for ti, tj in map_of_squares[i, j].forces:
                reversed_map[ti, tj].forces.add((i, j))
                reversed_map[i, j].forced_by.add((ti, tj))

    return reversed_map


def check_tiling_invariant(map_of_squares):
    """find_patches stage 4: a 2x2 block of blocked items must never happen - the
    alert_blocked/alert_chosen/forces bookkeeping from stages 2 and 3 exists
    specifically to prevent it. If it happens anyway, raise InvalidTilingError:
    that signals a bug upstream, not a recoverable case.
    """
    rows, cols = map_of_squares.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = (map_of_squares[i, j], map_of_squares[i, j + 1],
                       map_of_squares[i + 1, j], map_of_squares[i + 1, j + 1])
            if all(c.state == StateEnum.blocked for c in corners):
                raise InvalidTilingError(f"2x2 all-blocked block at ({i}, {j})")


def find_patches(map_of_squares):
    """Discover alert_chosen items, link them, and group them into patches.

    A "patch" is what we used to call a "graph" (and, briefly, a "path"): a
    connected group of alert_chosen items, linked together, that must all be
    chosen together. Patches - not single squares - are our real building block:
    "square placement" (place_square_in_core) is the low-level mechanical
    primitive, but only in the first couple of sublattice passes, before any
    seats have linked anything together, does a patch happen to consist of just
    one square. From here on we build the plane out of "patch placement".

    -- Data model (SquareItem / StateEnum) --
    map_of_squares is a padded 2D object array of SquareItem, one per block-position.
    Each item has:
      .quality        candidate score; -1 in the padding.
      .state          StateEnum.free / .chosen / .blocked.
      .alert_blocked  see stage 2 below.
      .alert_chosen   see stage 2 below.
      .forces  (row, col) index(es) of other item(s) this one is paired
                      with, or set() - see stages 2 and 3.
    Padding cells start out permanently StateEnum.blocked (quality < 0), so they can
    never be selected and the loops below only need a 1-cell margin of safety.

    -- Parallel placement, for context (place_square_in_core / test_square_placement
    / test_tiling) --
    The plane is covered by non-overlapping 3x3 cores on a 5-wide tile grid (1-cell
    border on each side). Only one sublattice of tiles - chosen by (row parity, col
    parity), 4 sublattices total - is active in any single simulated CUDA launch,
    because active tiles within one sublattice are 2 tiles (6 core-cells) apart, so no
    active core, nor its 1-cell diagonal write, ever touches another active tile's
    cells in the same pass. That is what lets place_square_in_core write
    .state = chosen/blocked directly, without needing to serialize across tiles.

    -- What find_patches resolves after a sweep --
    Direct writes keep the 4 sublattice passes conflict-free, but afterwards the plane
    can still be in a state no single placement call could safely react to by itself.
    find_patches runs sequentially over the whole map to resolve that, in four stages,
    still written with that parallel/sequential split in mind:

    1. Promote or block isolated free cells.

    2. alert_blocked / alert_chosen (set_alert_blocked, set_alert_chosen,
       iter_alert_thirds): a seat is a 2x2 block with three items blocked and one
       free. For a free item, look at its 8 neighbours (direct + diagonal) in ring
       order (RING_OFFSETS). Each of the 4 possible 2x2 blocks touching the item is
       that item plus one run of 3 consecutive ring indices (QUADRANT_TRIPLES): two
       direct neighbours and the diagonal corner between them. If two of those three
       are already blocked and the third is still free, blocking this item would turn
       that block into a real seat - iter_alert_thirds yields exactly that free
       third corner. The two already-blocked can be either a ring-adjacent pair
       (direct+corner, or corner+direct) or the triple's two direct neighbours
       themselves (leaving the corner between them, though not ring-adjacent to
       either, as the free third). When that happens: set_alert_blocked raises
       .alert_blocked on the item under consideration, and set_alert_chosen raises
       .alert_chosen on the free completing corner, recording the pairing by
       adding the alert_chosen item's index to the alert_blocked item's .forces.

    3. resolve_chosen_link: choosing an item blocks its 4 diagonal neighbours
       (DIAGONAL_OFFSETS). For every free item, every diagonal neighbour that is
       itself alert_blocked (and already has a link from stage 2) is adopted: its
       link(s) are added to item.forces - not just for items stage 2 already
       flagged alert_chosen, but for *any* free item with a qualifying diagonal
       neighbour, which this stage itself then marks alert_chosen (see the "Why
       link_patches looks past find_alerts's own alert_chosen items" note on
       do_closure). The result is a direct link between two alert_chosen items.
       Meaning: choosing one of them blocks their shared alert_blocked neighbour,
       which by construction would immediately complete a seat at the *other*
       linked item - so linked alert_chosen items must eventually be chosen
       together. Every item's new forces are resolved from the map as it stood at
       the start of this stage and only applied once every free item has been
       resolved (same snapshot-then-apply discipline as find_central_patch_items
       and find_cycle_patches), since a diagonal neighbour can itself be an
       alert_chosen item whose own .forces this same stage appends to.

    4. Promote or block isolated free cells.
       
    5. Invariant check: a fully-blocked 2x2 block must never occur - stage 2/3's
       alert_blocked/alert_chosen/link bookkeeping exists specifically to prevent it.
       If it happens anyway, raise InvalidTilingError: that signals a bug upstream,
       not a recoverable case.

    This function does not assign patch_id (see find_central_patch_items for that) -
    it only discovers alert_chosen items and links them together. patch_id used to
    be called graph_id (a field that has since been removed from SquareItem, back
    when this concept was called a "graph"), and used to be computed here, by
    assign_graph_id, reconciling ids pairwise (via min()) across linked alert_chosen
    items - that approach wasn't up to the task, so the procedure was removed.
    patch_id (assigned by find_central_patch_items, seeded at each patch's
    terminal) is what actually identifies a patch now.

    -- Open question: does this actually bound how far a patch can spread? --
    A chain of linked alert_chosen items is, in principle, a coordination
    requirement that reaches across tile boundaries - exactly what the sublattice
    scheme above is designed to avoid. If these patches could grow without bound, that
    would defeat the whole point of the parallel/sequential split (parallel
    place_square_in_core sweeps, sequential find_patches passes): resolving one patch
    could end up requiring information from far outside the tile that started it,
    destroying parallelism.

    The working hypothesis is that running find_patches after every single sublattice
    placement - i.e. resolving seats immediately rather than letting them accumulate
    across passes - keeps every patch's spatial extension bounded by a small constant
    (something like ten cells), regardless of how large the overall plane is. That is
    only a hypothesis right now, not something derived above. test_graph_spread (with
    eval_graphs, still unimplemented) is meant to measure it empirically first -
    collect the patches that form under repeated closure and look at
    max_graph_extension / max_num_nodes - before attempting an actual proof.
    """
    fill_isolated_free_cells(map_of_squares)
    find_alerts(map_of_squares)
    link_patches(map_of_squares)
    fill_isolated_free_cells(map_of_squares)
    check_tiling_invariant(map_of_squares)


def mark_patch_conflicts(map_of_squares):
    """For every alert_chosen item (patch_id assumed final - run find_patches, then
    find_central_patch_items to convergence, first), check the 4 diagonal
    neighbours it would block if chosen. A diagonal neighbour that is itself
    alert_chosen with a *different* patch_id means choosing either patch would
    block a member of the other - the two patches exclude each other, so each item
    records the other's patch_id in .conflicts (see SquareItem.conflicts and
    alert_graphs.mark_patch_conflicts, which does the actual per-item work). A
    diagonal neighbour sharing the *same* patch_id would mean the patch blocks its
    own member, which should never happen - alert_graphs.mark_patch_conflicts
    raises InvalidTilingError if it does.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.alert_chosen:
                mark_item_patch_conflicts(item, i, j, map_of_squares)


def find_central_patch_items(map_of_squares, gen):
    """Assign centrality to alert_chosen items, one generation at a time.

    gen counts how many times this function has already been called for
    map_of_squares: call it with gen=0 first, then gen=1, 2, ... on each subsequent
    call. Requires find_patches to have already run (.alert_chosen/.forces must be set).

    The function runs over all alert_chosen item. If gen=0 and the current item has a linked item
    with no .forces of its own, the linked item is a terminal item, and we set its centrality to 0,
    and its patch_id to its own flattened index (i * rows + j) - a fresh id for the patch that
    starts there.

    Each later call looks at every alert_chosen item that has a link: if the linked
    item's centrality is gen-1, the current item's centrality becomes gen. So
    centrality ends up measuring distance, in .forces hops, from a terminal item
    outward, one ring further with each generation.

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
    additionally checks its own .forces target: if that target now holds a *bigger*
    patch_id than the item itself has, the item adopts it too. Doing this every call
    lets a corrected id travel back down the losing patch, one hop per call, until
    the whole patch shares the same, single id - which is what makes patch_id
    double as a stable per-patch identifier in the end, without needing a separate
    pass to compute one: no additional "kernel calls" beyond the ones this function
    already needs for centrality.

    .conflicts propagates outward the same way, in that same ongoing check: every
    item copies over any entry from its .forces target's .conflicts that it doesn't
    already have. mark_patch_conflicts only ever writes a conflict at the specific
    item whose diagonal neighbour caused it, which can be anywhere in the patch, not
    just near the terminal - and unlike patch_id, that can happen at any time, not
    only once per item, so this can't be folded into the one-shot primary branch
    above the way patch_id's first assignment is; it has to be an ongoing check like
    this correction pass, run every call, so a conflict deposited late still makes
    its way outward. Moving strictly from lower centrality to higher, one hop per
    call, means every item's .conflicts is the union of everything found at or
    below it so far - so by the time an item reaches a patch's maximum centrality
    (its pivot - see closure.map_patches_to_pivots), its .conflicts holds every
    conflict found anywhere on the path from the terminal up to it. (It does *not*
    automatically include conflicts found only on a sibling branch through a
    different fan-in predecessor - see the "Open question" note on do_closure.)

    Reads for both correction steps are all taken before any of their writes are
    applied (same snapshot-then-apply discipline as find_cycle_patches), so a
    correction - patch_id or conflicts - can only move one hop per call, regardless
    of iteration order.

    Returns found: True if this call assigned a centrality, or propagated a
    patch_id or a conflicts correction, to at least one item, False otherwise -
    callers can loop, incrementing gen, until found comes back False to know
    everything reachable by .forces chains has fully converged.
    """
    rows, cols = map_of_squares.shape
    found = False
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.forces):
                continue
            target_pos = next(iter(item.forces))
            linked = map_of_squares[target_pos]
            if gen == 0:
                if not linked.forces:
                    linked.centrality = 0
                    li, lj = target_pos
                    linked.patch_id = li * rows + lj
                    found = True
            elif linked.centrality == gen - 1:
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

    corrections = []
    conflict_updates = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.forces):
                continue
            linked = map_of_squares[next(iter(item.forces))]

            if item.patch_id != -1 and linked.patch_id > item.patch_id:
                corrections.append((item, linked.patch_id))

            new_conflicts = linked.conflicts - item.conflicts
            if new_conflicts:
                conflict_updates.append((item, new_conflicts))

    for target, value in corrections:
        target.patch_id = value
    if corrections:
        found = True

    for target, new_conflicts in conflict_updates:
        target.conflicts.update(new_conflicts)
    if conflict_updates:
        found = True

    return found

def find_cycle_patches(map_of_squares, gen):
    """Ring leader election: find a parallel-safe break point for pure rings -
    patches with no terminal anywhere, so find_central_patch_items never assigns
    them a real centrality or patch_id (see the "Open question" note on do_closure).

    Every link is treated identically here - there's no reliance on visit order or
    which item happens to run first, only on each item's unique_id: its own
    flattened index (i * rows + j), the same convention find_central_patch_items
    uses to seed a terminal's patch_id.

    -- Known gap: fan-in candidates can still crowd each other out --
    More than one item can point at the same target via the (arbitrary) .forces
    entry this function reads - fan-in, the same shape link_patches's own .forces
    exists to support (see test_4links_situation) - so more than one candidate id
    can arrive at the same
    node in the same generation (test_do_closure_steps: (2, 2) receives a
    candidate from each of (2, 4), (3, 3), (4, 2), and (4, 4) at once, but only
    (3, 3) is actually on (2, 2)'s ring - the other three are tail branches that
    merely feed into it). max_id is a single scalar, so only one of several
    simultaneous candidates survives each generation - a list-based fix (one
    entry per still-live candidate) was tried and works, but was reverted in
    favour of the copy_map_reverse workaround (see its docstring and
    test_do_closure_steps_reverse_check) to avoid the extra bookkeeping. This
    scalar version is still subject to the crowding-out this describes; not
    fixed here.

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

def do_closure(map_of_squares):
    """Sequential clean-up step run after each parallel placement sweep.

    Thin wrapper around the two stages of closure: find_patches discovers alert_chosen
    items, links them, and groups them into patches (see its docstring for the full
    five-stage breakdown); mark_patch_conflicts then works out, for the now-final
    patches, which pairs of patches exclude each other. Both are exposed as standalone
    functions so callers - e.g. test_graph_spread - can invoke find_patches on its own.

    -- Why this exists at all: closure is what keeps an avalanche from building up --
    The whole point of running closure after every sweep, rather than only ever
    placing squares and reacting to problems as they're discovered, is to never leave
    the plane in a metastable state: one that looks fine (no invariant is currently
    violated) but that some single future placement could detonate into a much larger,
    unplanned cascade of forced consequences. An alert_blocked item sitting there
    unresolved is exactly such a stored-up cascade waiting for a trigger - closure's
    job is to find every one of those triggers now, while they're still cheap to
    reason about, and fold them into patches that get placed deliberately together,
    rather than being discovered piecemeal later as an "avalanche" of forced
    placements the parallel sweep can't safely absorb (see the "Parallel placement"
    note on find_patches for why that would be a problem: a forced placement
    triggered mid-avalanche could reach outside the tile a sweep is allowed to
    touch).

    -- Every free item diagonal to an alert_blocked item is a real trigger --
    *Any* free item diagonal to an alert_blocked item is a real trigger, not only
    the literal free corner of the one specific 2x2 block that made that item
    alert_blocked in the first place: choosing it blocks that neighbour exactly the
    same way, regardless of which quadrant originally made the neighbour
    alert_blocked (see test_4links_situation: an item with four alert_blocked
    diagonal neighbours, none of which happens to be the literal corner of any one
    of their own quadrants). Left undetected, a cell like that is precisely the
    metastable trap described above. set_alert_chosen (see its docstring) is what
    catches this - every free diagonal neighbour of an alert_blocked centre is
    linked directly, in the same single pass that discovers the centre is
    alert_blocked, so a cell's full consequences are captured immediately rather
    than left to detonate later. (This used to be link_patches's job, checking
    every free item after the fact for a diagonal alert_blocked neighbour it had
    missed - now unnecessary, since set_alert_chosen no longer misses it in the
    first place; see resolve_chosen_link's docstring.)

    -- Open question: do pivots ever miss a *shorter* branch's conflicts? --
    find_central_patch_items propagates .conflicts strictly from lower centrality to
    higher, one .forces hop at a time - so an item's .conflicts is the union of
    everything found on the *one* path from the terminal up to it, not necessarily
    everything found anywhere in the patch. map_patches_to_pivots closes the gap for
    items tied at the patch's own maximum centrality (see its docstring), by merging
    every tied peer's .conflicts into the chosen pivot before deciding whether the
    patch conflicts with anything. It does *not* reach a branch that dead-ends
    *before* reaching that maximum - a shorter leaf, with nothing tied to it, whose
    own conflicts (if it has any of its own, not shared with anything past it) never
    propagate anywhere once that leaf stops. Not yet checked whether this actually
    happens in practice, or whether it's ruled out some other way - flagged here,
    not fixed.
    """
    find_patches(map_of_squares)
    mark_patch_conflicts(map_of_squares)

def resolve_cycles_and_centrality(m, max_gens=20):
    """Run find_cycle_patches then find_central_patch_items to convergence on m,
    in place. 
    
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


# --- Pivots and cores --------------------------------------------------------
#
# Squares are placed one CUDA-style core (a 3x3 tile-core, see place_square_in_core
# and test_square_placement/test_tiling) at a time, and every item belongs to
# exactly one core. A patch, once fully grown, can be represented by a single one
# of its own items: a "pivot" - an item at the patch's own maximum centrality
# (there can be more than one; see find_central_patch_items's "Open question" note
# on do_closure - any one of them is picked, arbitrarily). Mapping a patch to its
# pivot's core means every patch belongs to exactly one core too, and - since a
# kernel call only ever chooses one square per core - exactly one patch is ever
# chosen per core in a given call. So the only conflicts that matter between two
# patches placed in the *same* kernel call are conflicts between patches in
# *different, non-neighbouring* cores - conflicts between patches sharing a core
# never come up, since only one of them could ever be chosen there anyway. That
# cuts the number of conflicts actually worth tracking by a lot.
#
# To exchange that information between cores rather than between every individual
# item, each core gets a single CoreItem (see map_of_squares.CoreItem) summarising
# every conflict that matters for patches represented in that core - build_core_map
# builds the full grid of them, one per tile-core, from the per-item .conflicts
# that find_central_patch_items has already grown up to each pivot.

def map_patches_to_pivots(map_of_squares, sz_border, sz_core):
    """Map every non-conflicting patch to a pivot item and the core it belongs to.

    A patch's pivot is one of its maximum-centrality items - if more than one is
    tied for maximum (its "peers"), one is picked arbitrarily (at random; which
    one doesn't matter, see the note above) - but first every peer's .conflicts is
    folded into the chosen pivot's. That matters because find_central_patch_items
    only ever propagates .conflicts along the *one* path from the terminal to a
    given item (see its docstring): a tied peer sitting on a different branch, through
    a different fan-in predecessor, can hold a conflict the pivot's own branch never
    saw. Merging every peer in first means the pivot ends up with the union of
    every conflict found on any branch that reaches the patch's own maximum
    centrality - whichever peer happens to get picked as pivot, it sees the same,
    complete set. (This does not reach conflicts found on a branch that dead-ends
    *before* the maximum centrality, i.e. a shorter leaf - see the "Open question"
    note on do_closure.)

    Conflicting patches - ones whose pivot still has a non-empty .conflicts once
    that merge is done - are left out for now (mapping those is deferred).

    Requires find_central_patch_items to have converged (every reachable item has
    a centrality and patch_id) and mark_patch_conflicts to have already run.

    Returns {patch_id: (pivot_position, core_index)}, where pivot_position is the
    pivot's (row, col) in map_of_squares and core_index is the (I, J) tile-core it
    falls in, using the same (sz_border, sz_core) convention as
    test_square_placement/test_tiling.
    """
    rows, cols = map_of_squares.shape
    members_by_patch = {}
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if item.alert_chosen and item.patch_id != -1:
                members_by_patch.setdefault(item.patch_id, []).append((i, j, item.centrality))

    pivots = {}
    for patch_id, members in members_by_patch.items():
        max_centrality = max(centrality for _, _, centrality in members)
        candidates = [(i, j) for i, j, centrality in members if centrality == max_centrality]
        pivot_i, pivot_j = random.choice(candidates)
        pivot_item = map_of_squares[pivot_i, pivot_j]

        for peer_i, peer_j in candidates:
            if (peer_i, peer_j) == (pivot_i, pivot_j):
                continue
            pivot_item.conflicts.update(map_of_squares[peer_i, peer_j].conflicts)

        if pivot_item.conflicts:
            continue  # conflicting patch - put aside for later

        core_I = (pivot_i - sz_border) // sz_core
        core_J = (pivot_j - sz_border) // sz_core
        pivots[patch_id] = ((pivot_i, pivot_j), (core_I, core_J))
    return pivots


def build_core_map(map_of_squares, sz_border, sz_core, num_tiles_v, num_tiles_h):
    """Build the core map: one CoreItem per 3x3 tile-core (see map_of_squares.CoreItem).

    A core's CoreItem.conflicts is the union of every one of its 9 items'
    .conflicts, with any patch_id that is *also* one of those same 9 items' own
    .patch_id discarded - an intra-core conflict never matters, since only one
    patch is ever chosen per core to begin with (see the note above).
    """
    core_map = np.empty((num_tiles_v, num_tiles_h), dtype=object)
    for I in range(num_tiles_v):
        for J in range(num_tiles_h):
            row0 = sz_border + I * sz_core
            col0 = sz_border + J * sz_core

            local_patch_ids = set()
            merged_conflicts = set()
            for di in range(sz_core):
                for dj in range(sz_core):
                    item = map_of_squares[row0 + di, col0 + dj]
                    if item.patch_id != -1:
                        local_patch_ids.add(item.patch_id)
                    merged_conflicts.update(item.conflicts)

            core_item = CoreItem()
            core_item.conflicts = list(merged_conflicts - local_patch_ids)
            core_map[I, J] = core_item
    return core_map
