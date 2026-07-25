import random

import numpy as np

from map_of_squares import StateEnum, InvalidTilingError, CoreItem
from alert_graphs import (RING_OFFSETS,
                           set_alert_blocked,
                           set_alert_chosen,
                           link_chosen_items,
                           mark_patch_conflicts as mark_item_patch_conflicts)


def promote_isolated_free_cells(map_of_squares):
    """find_patches stage 1: promote any free cell whose 4 direct (orthogonal)
    neighbours are all blocked to chosen. Such a cell can never overlap a chosen
    square, so placing it is always safe - and it is the only way it will ever get
    covered, since no future core sweep would treat it as reachable.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            neighbours = (map_of_squares[i - 1, j], map_of_squares[i + 1, j],
                          map_of_squares[i, j - 1], map_of_squares[i, j + 1])
            if all(n.state == StateEnum.blocked for n in neighbours):
                item.state = StateEnum.chosen


def find_alerts(map_of_squares):
    """find_patches stage 2 (set_alert_blocked, set_alert_chosen, iter_alert_thirds):
    an "alert" is a 2x2 block with three items blocked and one free. For a free
    item, look at its 8 neighbours (direct + diagonal) in ring order (RING_OFFSETS).
    Each of the 4 possible 2x2 blocks touching the item corresponds to one run of 3
    consecutive ring indices (QUADRANT_TRIPLES): two direct neighbours and the
    diagonal between them. If two ring-adjacent neighbours are blocked *and* the
    third (completing) corner of that block is still free, blocking this item
    would turn that block into a real alert - iter_alert_thirds yields exactly
    those completing corners. When that happens: set_alert_blocked raises
    .alert_blocked on the item under consideration, and set_alert_chosen raises
    .alert_chosen on the free completing corner, recording the pairing by pointing
    the alert_blocked item's .link at the alert_chosen item's index.
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
                set_alert_chosen(item, i, j, ring)


def link_patches(map_of_squares):
    """find_patches stage 3 (link_chosen_items): choosing an item blocks its 4
    diagonal neighbours (DIAGONAL_OFFSETS). For every alert_chosen item, if one of
    those diagonal neighbours is itself alert_blocked (and already has a .link
    from stage 2), that link is adopted: item.link = neighbour.link. The result is
    a direct link between two alert_chosen items. Meaning: choosing one of them
    blocks their shared alert_blocked neighbour, which by construction would
    immediately create an alert at the *other* linked item - so linked
    alert_chosen items must eventually be chosen together.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.alert_chosen:
                link_chosen_items(item, i, j, map_of_squares)


def check_tiling_invariant(map_of_squares):
    """find_patches stage 4: a 2x2 block of blocked items must never happen - the
    alert_blocked/alert_chosen/link bookkeeping from stages 2 and 3 exists
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
    alerts have linked anything together, does a patch happen to consist of just
    one square. From here on we build the plane out of "patch placement".

    -- Data model (SquareItem / StateEnum) --
    map_of_squares is a padded 2D object array of SquareItem, one per block-position.
    Each item has:
      .quality        candidate score (from image_squares_ranked); -1 in the padding.
      .state          StateEnum.free / .chosen / .blocked.
      .alert_blocked  see stage 2 below.
      .alert_chosen   see stage 2 below.
      .link           (row, col) index of another item this one is paired with, or
                       None - see stages 2 and 3.
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

    1. Promote any free cell whose 4 direct (orthogonal) neighbours are all blocked to
       chosen. Such a cell can never overlap a chosen square, so placing it is always
       safe - and it is the only way it will ever get covered, since no future core
       sweep would treat it as reachable.

    2. alert_blocked / alert_chosen (set_alert_blocked, set_alert_chosen,
       iter_alert_thirds): an "alert" is a 2x2 block with three items blocked and one
       free. For a free item, look at its 8 neighbours (direct + diagonal) in ring
       order (RING_OFFSETS). Each of the 4 possible 2x2 blocks touching the item
       corresponds to one run of 3 consecutive ring indices (QUADRANT_TRIPLES): two
       direct neighbours and the diagonal between them. If two ring-adjacent
       neighbours are blocked *and* the third (completing) corner of that block is
       still free, blocking this item would turn that block into a real alert -
       iter_alert_thirds yields exactly those completing corners. When that happens:
       set_alert_blocked raises .alert_blocked on the item under consideration, and
       set_alert_chosen raises .alert_chosen on the free completing corner, recording
       the pairing by pointing the alert_blocked item's .link at the alert_chosen
       item's index.

    3. link_chosen_items: choosing an item blocks its 4 diagonal neighbours
       (DIAGONAL_OFFSETS). For every alert_chosen item, if one of those diagonal
       neighbours is itself alert_blocked (and already has a .link from stage 2), that
       link is adopted: item.link = neighbour.link. The result is a direct link
       between two alert_chosen items. Meaning: choosing one of them blocks their
       shared alert_blocked neighbour, which by construction would immediately create
       an alert at the *other* linked item - so linked alert_chosen items must
       eventually be chosen together.

    4. Invariant check: a fully-blocked 2x2 block must never occur - stage 2/3's
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
    placement - i.e. resolving alerts immediately rather than letting them accumulate
    across passes - keeps every patch's spatial extension bounded by a small constant
    (something like ten cells), regardless of how large the overall plane is. That is
    only a hypothesis right now, not something derived above. test_graph_spread (with
    eval_graphs, still unimplemented) is meant to measure it empirically first -
    collect the patches that form under repeated closure and look at
    max_graph_extension / max_num_nodes - before attempting an actual proof.
    """
    promote_isolated_free_cells(map_of_squares)
    find_alerts(map_of_squares)
    link_patches(map_of_squares)
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
    call. Requires find_patches to have already run (.alert_chosen/.link must be set).

    The function runs over all alert_chosen item. If gen=0 and the current item has a linked item
    with no .link of its own, the linked item is a terminal item, and we set its centrality to 0,
    and its patch_id to its own flattened index (i * rows + j) - a fresh id for the patch that
    starts there.

    Each later call looks at every alert_chosen item that has a link: if the linked
    item's centrality is gen-1, the current item's centrality becomes gen. So
    centrality ends up measuring distance, in .link hops, from a terminal item
    outward, one ring further with each generation.

    In that same step, patch_id propagates outward alongside centrality: the linked
    item (centrality gen-1, closer to the terminal) hands its patch_id to the current
    item (centrality gen, one step further out) -
      - if the current item has no patch_id yet (-1), it adopts the linked item's,
      - if it already has a *different* one (two patches have merged here), it keeps
        whichever of the two is larger - an arbitrary but consistent tie-break,
      - if it already has the *same* one, this item has already been reached by this
        patch once before - the only way that happens is by looping back around a
        cycle - so instead of reassigning, the current item's own .link is cut
        (set to None) to stop the patch from being retraced forever.

    A merge like that only fixes the patch_id of the item sitting at the merge point
    itself - everything closer to the terminal on the losing patch was already given
    the smaller id in some earlier call, before the merge was even discovered, and
    that's stale now. So on every call, every item that already has a patch_id
    additionally checks its own .link target: if that target now holds a *bigger*
    patch_id than the item itself has, the item adopts it too. Doing this every call
    lets a corrected id travel back down the losing patch, one hop per call, until
    the whole patch shares the same, single id - which is what makes patch_id
    double as a stable per-patch identifier in the end, without needing a separate
    pass to compute one: no additional "kernel calls" beyond the ones this function
    already needs for centrality.

    .conflicts propagates outward the same way, in that same ongoing check: every
    item copies over any entry from its .link target's .conflicts that it doesn't
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
    everything reachable by .link chains has fully converged.
    """
    rows, cols = map_of_squares.shape
    found = False
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.link is not None):
                continue
            linked = map_of_squares[item.link]
            if gen == 0:
                if linked.link is None:
                    linked.centrality = 0
                    li, lj = item.link
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
                    item.link = None

    corrections = []
    conflict_updates = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.link is not None):
                continue
            linked = map_of_squares[item.link]

            if item.patch_id != -1 and linked.patch_id > item.patch_id:
                corrections.append((item, linked.patch_id))

            new_conflicts = [c for c in linked.conflicts if c not in item.conflicts]
            if new_conflicts:
                conflict_updates.append((item, new_conflicts))

    for target, value in corrections:
        target.patch_id = value
    if corrections:
        found = True

    for target, new_conflicts in conflict_updates:
        for c in new_conflicts:
            target.conflicts.append(c)
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

    All reads below see map_of_squares exactly as it was at the start of this call
    - writes are collected and only applied once every edge has been evaluated, the
    same synchronisation a genuinely parallel pass over the ring would need before
    moving on to the next step. That's what makes "after m steps [m = ring size]
    there's only one [survivor]" a real guarantee: a candidate can move at most one
    edge per call, regardless of iteration order.

    For the current item A (a = A's unique_id, at position (i, j)) and its linked
    item B (B = map_of_squares[A.link]):

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
          can ever come back around - so this is the leader, confirmed. A.link is
          removed, opening the ring right here: A becomes an ordinary terminal
          (something points at it, it points at nothing), which is exactly what
          find_central_patch_items needs to seed centrality/patch_id from - once
          called again after the ring has been opened.
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
            if not (A.alert_chosen and A.link is not None):
                continue
            B = map_of_squares[A.link]

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
                cuts.append(A)
            else:  # A.max_id > a
                resets.append(A)
                propagations.append((B, A.max_id))

    for target in resets:
        target.max_id = -1
    for target, value in seeds:
        target.max_id = value
    for target in cuts:
        target.link = None
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

    -- Open question: do pivots ever miss a *shorter* branch's conflicts? --
    find_central_patch_items propagates .conflicts strictly from lower centrality to
    higher, one .link hop at a time - so an item's .conflicts is the union of
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
    that merge is done - are left out for now (mapping those is deferred; see
    do_closure's docstring and the "Alert resolution" note below).

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
            for c in map_of_squares[peer_i, peer_j].conflicts:
                if c not in pivot_item.conflicts:
                    pivot_item.conflicts.append(c)

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
            merged_conflicts = []
            for di in range(sz_core):
                for dj in range(sz_core):
                    item = map_of_squares[row0 + di, col0 + dj]
                    if item.patch_id != -1:
                        local_patch_ids.add(item.patch_id)
                    for c in item.conflicts:
                        if c not in merged_conflicts:
                            merged_conflicts.append(c)

            core_item = CoreItem()
            core_item.conflicts = [c for c in merged_conflicts if c not in local_patch_ids]
            core_map[I, J] = core_item
    return core_map


# --- Alert resolution -------------------------------------------------------
#
# Placing a square S can raise alerts nearby (see find_patches). Every alert_chosen
# item belongs to a patch_id-identified patch - a connected, variable-shaped group
# of alert_chosen items that is our actual building block once alerts start
# linking squares together (see the note at the top of find_patches). Some pairs
# of patches conflict with each other (mark_patch_conflicts): choosing every item
# in one would block an item the other patch needs, so the two can never both be
# placed.
#
# Q(S) is the set of every way to pick a combination of mutually non-conflicting
# patches relevant to S: an element q of Q(S) is itself a set of patch_ids, no two
# of which appear in each other's .conflicts. The set of squares belonging to q is
# the union, over every patch_id in q, of that patch's alert_chosen items - i.e.
# everything that would additionally need placing if q's patches are chosen
# alongside S.
#
# The alert-resolution ("patch placement") procedure for one just-placed square S
# is then:
#   1. find Q(S)
#   2. choose one q in Q(S), by the desirability of q's square set
#   3. place S and every square in q
#
# The point, and the reason this whole module is called "closure": placing S
# obliges placing every square in q, but placing q obliges nothing beyond q
# itself - nothing in q's own square set raises an alert that reaches outside q.
# That's the same relationship a closed set in topology has to its limit points:
# q already contains everything its own placement would otherwise force in.
#
# Everything above this point (find_patches, mark_patch_conflicts,
# find_central_patch_items, find_cycle_patches, map_patches_to_pivots,
# build_core_map) discovers and describes the patches and their conflicts. The
# functions below are where that description turns into an actual decision and
# placement - stubs for now, to be fleshed out later.

def find_patch_combinations(map_of_squares, S):
    """Find Q(S): every conflict-free combination of alert patches relevant to
    placing S - i.e. every set q of patch_ids such that no two patch_ids in q
    appear in each other's .conflicts (see mark_patch_conflicts).

    S identifies the square being placed - exact form (position, or the
    SquareItem itself) not decided yet.

    Returns Q, a set (or list) of q's - each q itself a set of patch_id.
    """
    pass


def squares_in_combination(map_of_squares, q):
    """The set of alert_chosen item positions belonging to combination q: the
    union, over every patch_id in q, of that patch's alert_chosen items - i.e.
    everything that would need to be placed alongside S if q is chosen.
    """
    pass


def choose_combination(map_of_squares, S, Q):
    """Pick one q out of Q(S) (see find_patch_combinations), by the desirability
    of the square set it would add (squares_in_combination(map_of_squares, q)).
    What "desirability" means - total quality, size, something else - is not
    decided yet.
    """
    pass


def place_closure(map_of_squares, S, q):
    """Commit S and every square in q (see squares_in_combination) to
    StateEnum.chosen. The actual placement step, once find_patch_combinations and
    choose_combination have picked q for S.
    """
    pass


def resolve_square_closure(map_of_squares, S):
    """The full alert-resolution ("patch placement") procedure for one just-placed
    square S - see the "Alert resolution" note above this function for the
    concepts involved:

    1. find_patch_combinations: find Q(S).
    2. choose_combination: pick one q in Q(S).
    3. place_closure: place S and every square in q.
    """
    pass
