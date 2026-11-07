"""The condition-based closure algorithm itself - as opposed to conditions.py,
which only builds the vacuum state (vacuum_condition_offsets, initial_map, on
a plain map_of_squares.SquareItem array) for this algorithm to run on. This is
the do_closure2 counterpart
to do_closure/do_closure_intern's own pipeline (closure.py) - see
squareplacement_superlattice_refactor memory note for why that pipeline is
being replaced rather than extended.
"""
from functools import lru_cache
from itertools import combinations, product

from map_of_squares import StateEnum, PlacementStatus
from cores import in_realm


def _has_diagonal_conflict(term):
    """True if any two ids in term are diagonal neighbours of each other -
    two cells that can never both be chosen (choosing either one diagonally
    blocks the other), so a term containing such a pair describes a
    combination that could never actually happen. vacuum_condition_offsets'
    own generator already excludes this *within* one original triple; this
    is the same check, used by compute_eligibility's flattening step to
    catch it *across* triples too, which nothing else checks for.

    term's own elements are (id, placement_status) pairs (see SquareItem.
    forced_by's own docstring) - only the id half matters for this geometric
    check, placement_status is ignored here.
    """
    return any(abs(a_id[0] - b_id[0]) == 1 and abs(a_id[1] - b_id[1]) == 1
               for (a_id, _), (b_id, _) in combinations(term, 2))


@lru_cache(maxsize=None)
def _forces_offsets():
    """The grouping half of init_vacuum's own forces(id_) computation,
    worked out once at the origin (0, 0) rather than once per id_ - see
    init_vacuum's own docstring for the underlying (T, c) -> B = -c logic;
    this is exactly that, with id_ fixed at (0, 0), so every B and every
    condition member here are themselves still plain offsets, not absolute
    positions yet.

    Translation-invariant, same reasoning as vacuum_condition_offsets' own
    caching: this grouping (which b_offsets exist, and which condition
    offsets belong to each) never depends on id_, only the final translation
    step does - so the expensive part (building the by-B groups from 228 raw
    (T, c) pairs) happens once, cached, and init_vacuum's own per-id_ work
    reduces to translating the already-grouped result, not rebuilding the
    grouping from scratch every time.

    Returns a set of (b_offset, condition_offsets) pairs, condition_offsets
    itself a frozenset of frozensets of (di, dj) offsets - the same shape
    forces(id_) has, just not yet translated by id_.
    """
    from conditions import vacuum_condition_offsets
    offsets = vacuum_condition_offsets()
    by_b_offset = {}
    for triple in offsets:
        for di, dj in triple:
            b_offset = (-di, -dj)
            condition_offset = frozenset((b_offset[0] + d2i, b_offset[1] + d2j)
                                          for d2i, d2j in triple)
            by_b_offset.setdefault(b_offset, set()).add(condition_offset)
    return {(b_offset, frozenset(condition_offsets))
            for b_offset, condition_offsets in by_b_offset.items()}


def init_vacuum(id_):
    """Compute (forced_by, forces) for id_ straight from the universal vacuum
    offset structure - id_'s own position is the only input, no other cell
    and no board shape involved.

    In the vacuum, every term is of the form (id_+c1, id_+c2, id_+c3) for one
    of conditions.vacuum_condition_offsets' own constant offset triples
    (c1, c2, c3) - forced_by(id_) is just that translation applied to every
    one of the 76 triples. forces(id_) is the same structure read backwards:
    id_ is a member of some other cell B's own forced_by triple T exactly
    when id_ = B + c for one of T's own offsets c, i.e. B = id_ - c - so
    every (T, c) pair contributes one (B, condition) entry, grouped by B
    (merged into one set when more than one (T, c) pair lands on the same B).
    That grouping is translation-invariant, so it's done once, at the origin,
    by _forces_offsets (cached) - forces(id_) here is just that cached result
    translated by id_, not rebuilt from the raw 228 (T, c) pairs every call.
    Confirmed against the full-board, cross-cell computation (initial_map +
    update_forces) for an interior id, and against the pre-caching version of
    this same function - identical sets every way.

    This is the unbounded, infinite-board version - it knows nothing about a
    real board's edges. A caller assembling an actual finite board (see
    conditions.initial_map) still has to filter out any term that would need
    an off-board id - same rule as before, drop the whole triple rather than
    just the off-board member (see initial_map's own docstring for why that
    matters) - init_vacuum does not do this filtering itself.

    Returns (forced_by, forces) as plain data - nothing here writes to any
    map; the caller decides whether and where to store them.
    """
    # local import: conditions.py imports bruteforce.update_forces at module
    # level, so importing conditions back at this module's own top level
    # would cycle - deferring to call time (by when conditions.py is already
    # fully loaded) sidesteps it, same trick closure.py's _call_step_asserts
    # uses for test_utils.DoClosureSteps.
    from conditions import vacuum_condition_offsets
    offsets = vacuum_condition_offsets()

    forced_by = {frozenset((id_[0] + di, id_[1] + dj) for di, dj in triple)
                 for triple in offsets}

    forces = set()
    for b_offset, condition_offsets in _forces_offsets():
        b = (id_[0] + b_offset[0], id_[1] + b_offset[1])
        conditions = frozenset(
            frozenset((id_[0] + di, id_[1] + dj) for di, dj in condition_offset)
            for condition_offset in condition_offsets)
        forces.add((b, conditions))

    return forced_by, forces


def update_forces(m):
    """Recompute every cell's .forces from the current .forced_by, in place -
    the reverse-index refresh SquareItem.forces's own docstring says nothing
    keeps automatic. Call this after anything that mutates .forced_by
    (compute_conditions, compute_eligibility both do, and both call this
    themselves at the end - see their own docstrings) if .forces needs to
    stay consistent with it.

    Full recompute, not incremental: every cell's .forces is cleared first,
    then rebuilt exactly as conditions.initial_map's own first build does -
    for every b and every one of its current .forced_by conditions, every
    member id a of that condition gets (b, conditions) added to a.forces
    (indexed by the plain id, not the (id, placement_status) pair - see
    SquareItem.forced_by's own docstring; the condition itself, tags and
    all, is what gets stored, unchanged), merging b's qualifying conditions
    together into one set when the same b reaches a via more than one of its
    own triples.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            m[i, j].forces = set()

    by_cell = {}
    for i in range(rows):
        for j in range(cols):
            b = (i, j)
            for condition in m[b].forced_by:
                for a, _a_status in condition:
                    by_cell.setdefault(a, {}).setdefault(b, set()).add(condition)
    for a, forcing in by_cell.items():
        m[a].forces = {(b, frozenset(conditions)) for b, conditions in forcing.items()}


def retag_forced_by(m):
    """Refresh every .forced_by triple member's embedded placement_status to
    match that id's CURRENT m[id].placement_status, in place - same
    discipline as update_forces keeping .forces in sync with .forced_by (see
    its own docstring): nothing does this automatically when
    .placement_status changes elsewhere (e.g. test_context_tiles.
    mark_placement_cores, or compute_eligibility's own realm-relative
    overwrite), so call this explicitly afterward if the embedded tags need
    to reflect it. The ids themselves and the triple structure are
    untouched, only the status half of each (id, placement_status) pair is
    replaced. Calls update_forces(m) at the end, since .forces' own nested
    conditions are the same objects as .forced_by's - rebuilding .forced_by
    here makes them stale otherwise.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            m[i, j].forced_by = {
                frozenset((member_id, m[member_id].placement_status)
                          for member_id, _old_status in condition)
                for condition in m[i, j].forced_by
            }
    update_forces(m)


def core_index(id_, M, s=0):
    """Which (ci, cj) M x M core id_ falls in, given the core grid's own
    origin s - same (s + ci*M, s + cj*M) convention as test_context_tiles.
    build_framed_map/mark_placement_cores. Pure coordinate arithmetic on
    id_ alone, no map involved - works for any id, on-board or not.
    """
    row, col = id_
    return (row - s) // M, (col - s) // M


def prune_same_core_candidates(m, M, s=0):
    """Drop (not reduce) every .forced_by condition that has two or more
    members both PlacementStatus.candidate and in the same M x M core (see
    core_index), in place.

    A placement core only ever has one of its own cells actually chosen
    (test_context_tiles.mark_placement_cores' own discipline, the point of
    the placement/transit core split) - a condition demanding two candidate
    cells of the *same* core both be chosen could therefore never be
    satisfied, structurally, regardless of current .state. This is a
    different kind of reduction from compute_conditions' own - that one
    reacts to what has actually become chosen/blocked so far; this one is a
    standing structural fact, true from the moment the board's placement
    cores are fixed, independent of any placement that has happened yet.

    A non-free cell has nothing left to be forced into, so its own
    .forced_by is cleared outright - same convention as compute_conditions,
    and for the same reason: update_forces rebuilds .forces from every
    cell's current .forced_by, free or not, so a stale, un-pruned
    .forced_by left on a chosen/blocked cell would otherwise leak straight
    back into some other, free cell's own .forces entries.

    Mutates .forced_by, so update_forces(m) is called at the end to keep
    .forces consistent with it - see that function's own docstring.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if item.state != StateEnum.free:
                item.forced_by.clear()
                continue
            kept = set()
            for condition in item.forced_by:
                candidate_cores = [core_index(member_id, M, s)
                                   for member_id, status in condition
                                   if status == PlacementStatus.candidate]
                if len(candidate_cores) != len(set(candidate_cores)):
                    continue
                kept.add(condition)
            item.forced_by = kept
    update_forces(m)


def compute_conditions(m):
    """Drop or shrink every condition whose outcome has partly or fully
    settled, in place - the generalization, to n-ary conditions, of the old
    pipeline's own alert_chosen/alert_blocked reductions (reapplied here
    2026-10-22: "apply the old reductions").

    An element id is looked up as a cell index: m[id].state. A condition is
    false, and dropped entirely, if any of its elements is blocked - that
    placement can never happen. Otherwise, every element that is already
    chosen is removed from the condition (its own requirement is already
    satisfied, so it no longer needs tracking) - what remains is a smaller
    condition recording only what's still needed: a triple with one chosen
    element reduces to a pair, with two chosen elements reduces to a single.
    A single-element condition is exactly the old pipeline's own binary link
    (.forced_by1) as a special case - "we end up with something like our
    alerts, forces, forced_by, but more general" (the original design note
    this confirms). If every element was already chosen, the reduced
    condition is empty - the cell itself is forced, same as before, and the
    condition is dropped (not kept as an empty set). A condition with no
    blocked element and at least one element still free is kept, reduced to
    just its still-free elements.

    A cell that is already chosen or blocked has nothing left to be forced
    into, so all of its conditions are cleared.

    Mutates .forced_by, so update_forces(m) is called at the end to keep
    .forces consistent with it - see that function's own docstring.

    Returns the list of (row, col) positions of free cells that had a
    condition become fully true; placing them is left to the caller.
    """
    forced = []
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if item.state != StateEnum.free:
                item.forced_by.clear()
                continue
            became_true = False
            kept = set()
            for condition in item.forced_by:
                if any(m[pos].state == StateEnum.blocked for pos, _status in condition):
                    continue
                remaining = frozenset((pos, status) for pos, status in condition
                                       if m[pos].state != StateEnum.chosen)
                if not remaining:
                    became_true = True
                    continue
                kept.add(remaining)
            item.forced_by = kept
            if became_true:
                forced.append((i, j))
    update_forces(m)
    return forced


def unblock_path(m, id_):
    """Permanently block a self-blocking cell and recompute the board from
    the updated .state - the new architecture's own analog of the old
    pipeline's dissolve_blocked_paths + do_closure_intern's second-pass
    recompute (see squareplacement_superlattice_refactor memory note,
    2026-10-29 entry, "Old pipeline's own self-blocking handling recalled").
    Named "unblock_path" per Ivo's own instruction that day, even though
    the direct action is blocking one cell - the NAME describes the
    purpose it serves for every OTHER path that was needlessly constrained
    by a candidate that could never actually have been chosen: once id_ is
    confirmed permanently blocked, those other paths are freed/"unblocked"
    from a conflict that was never real.

    "Play it safe": rather than inventing any new, bespoke invalidation
    logic for whatever else happened to depend on id_, this does the
    minimum - set id_'s own .state, then call the SAME, already-trusted
    compute_conditions (see its own docstring) the rest of this module
    already relies on to propagate the consequence - "a condition is false,
    and removed, if any of its elements is blocked" already does exactly
    the "many paths converge into one cell, if that cell is self-blocking
    this blocks all of them" job map-wide, in one pass, with no separate
    per-path tracing needed - matching dissolve_blocked_paths's own
    original motivation precisely, just via this architecture's own
    machinery instead of .path_id/.forces1.

    id_ is expected to already be a confirmed self-blocking candidate - see
    the former cores.CoreItem.update_blocked (removed 2026-11-07): id_'s own forcing chain
    reaches a diagonal neighbour of itself, a genuine contradiction, so id_
    could never validly have been chosen in the first place. This function
    does not itself check that (callers are expected to have found id_ via
    .blocked's own (id_, id_) self-blocking pairs); it only applies the
    consequence.

    Mutates id_'s own .state (free -> blocked) and, via compute_conditions,
    every cell's own .forced_by/.forces map-wide - NOT any CoreItem's own
    .forced_by/.blocked, which are each their own, separate, explicit
    recompute (update_links) a caller runs afterward over whichever tile(s) the
    change actually reaches, same as after any other board mutation.
    """
    m[id_].state = StateEnum.blocked
    compute_conditions(m)


def collapse_synced_ids(m, id0, synced_ids):
    """Implemented 2026-11-01 (step 1 of Ivo's own base-level ordering,
    same day: "the case id1,id2 ... is identified with some id0, creating
    additional links"). Covers points 1 and 2 of Ivo's own downstream
    planning note (see squareplacement_superlattice_refactor memory note,
    2026-10-31): id0's own forcing chain forces every id in synced_ids
    (id1, id2, ... - discovered via the former CoreItem.update_blocked's own
    transitive rule, 2026-10-29, which already finds the id0/id1/id2
    triple this acts on). Two cases share this one mechanism:

    1. synced_ids also mutually exclude each other (the self-blocking case
       update_blocked already detects): id0 itself can then never validly
       be chosen - handled separately, by unblock_path (id0 is the id_ that
       gets permanently blocked there, not any of synced_ids).
    2. synced_ids do NOT mutually exclude: id0 being chosen forces every
       one of them together, with no contradiction - id0 and every synced
       id are now permanently linked, "choosing id1 without id2 has become
       impossible" (Ivo's own words) regardless of which specific id among
       them is actually placed.

    Either way, every id in synced_ids is now fully interchangeable with
    id0 as far as any OTHER condition/term on the board is concerned - its
    own separate identity carries no more information than id0's already
    does.

    Action: replace every occurrence of every id in synced_ids with id0
    itself, wherever it appears as a condition member in ANY SquareItem's
    own .forced_by, map-wide, re-tagging the replaced member's own
    embedded placement_status to id0's current one (same discipline as
    retag_forced_by) - a condition that named two or more of these ids as
    separate members collapses down to however many DISTINCT ids remain,
    for free, via ordinary frozenset deduplication (e.g. a triple naming
    {id1, id2, id3} collapses to the pair {id0, id3} once id1 and id2 both
    become id0) - "creating more links," Ivo's own words, since a smaller,
    more-reduced term is a stronger, more immediately actionable fact than
    the original triple ever was (same "n-aries just link paths together"
    mechanism already confirmed 2026-10-23, now triggered by an explicit
    rewrite instead of compute_conditions's own gradual one-member-at-a-
    time shrinking). The same dedup carries all the way down: if every
    member of a condition maps to id0 (e.g. a pair {id1, id2} both
    syncing), it collapses to a plain size-1 condition {id0} - "(id0, id0)
    becomes id0" (Ivo, 2026-11-01), the old pipeline's own single-link
    shape - with no special case needed, since the comprehension below
    produces the same (id0, status) tuple once per original member and
    frozenset() dedups identical tuples on its own. update_forces(m) is
    called at the end to keep .forces consistent with the now-simplified
    .forced_by - same discipline as every other .forced_by mutator in
    this file.

    Deliberately does NOT recompute the CoreItem pyramid (Ivo, same day:
    "Don't recompute the tile yet") - callers remain responsible for
    recomputing CoreItem.forced_by/.blocked (and update_aliases_down as
    relevant) on whichever tiles the change reaches, same
    discipline unblock_path already established; this function does not do
    that itself, same as unblock_path doesn't either.
    """
    replacement = {sid: id0 for sid in synced_ids}
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            item.forced_by = {
                frozenset(
                    (replacement.get(member_id, member_id),
                     m[replacement.get(member_id, member_id)].placement_status)
                    for member_id, _status in condition)
                for condition in item.forced_by
            }
    update_forces(m)


def create_compound_ids(ids):
    """Implemented 2026-11-01 (step 2 of Ivo's own base-level ordering,
    same day). Renamed from "virtual id" to "compound id" (Ivo, same day) -
    covers point 3 of Ivo's own downstream planning note (see
    squareplacement_superlattice_refactor memory note, 2026-10-31): ids
    (idx, idy, ... - at least 2) belong to DIFFERENT placement tiles, each
    independently choosable in the same round, with no forcing chain
    linking them (unlike collapse_synced_ids's own case) - so they CAN be
    chosen together, but don't have to be, and neither implies the other
    the way a forced id does. This makes any n-ary term naming two or more
    of them relevant/activatable once they are in fact chosen together,
    but WITHOUT collapsing their own separate identities away (Ivo's own
    words: "keeping id1 and id2") - unlike collapse_synced_ids, which does
    replace them.

    Changed from the earlier plan (2026-10-31): a compound id's own value
    is now arbitrary - only required to be unique, not a deterministic
    encoding of its own members (the earlier docstring's own
    "frozenset({id1, id2})" idea) - with the (compound_id, members) mapping
    tracked separately, in the returned compound_ids set, rather than
    folded into the id's own structure. Tagged ("compound", n) rather than
    a bare integer so a compound id can never collide with a real (row,
    col) cell id.

    For every non-trivial subset of size >= 2 of ids (n > 3 included -
    Ivo, 2026-11-01: "i still maintain n-aries with n>3 matter" - so for 3
    ids id1/id2/id3: the three pairs (id1,id2)/(id2,id3)/(id1,id3), plus
    the full triple (id1,id2,id3), matching Ivo's own worked example),
    creates one fresh compound id standing in for "this specific subset,
    all chosen together."

    Returns compound_ids, a set of (compound_id, members, is_subset_of)
    triples (expanded from the original (compound_id, members) pairs,
    Ivo, 2026-11-01) - members itself a tuple, the subset this compound_id
    stands for; is_subset_of a frozenset of every OTHER compound_id whose
    own members is a superset of this entry's own members (the compound
    ids that "contain A as a subset"), computed only when len(members) <=
    3 - above that, is_subset_of is always the empty frozenset, Ivo's own
    instruction ("Do this only for A with number of id1,id2... <=3. Empty
    otherwise") - matching the standing "a real board condition is capped
    at 3 members" fact: only a size-<=3 compound id could ever actually
    substitute into one, so only those need to know what bigger compound
    id(s) they're also a part of. No restriction on the SIZE of the
    containing compound id itself - a size-2 or size-3 A can be recorded
    as a subset of a B with more than 3 members.

    Deliberately does nothing beyond building this set (Ivo, same day,
    twice: "I will tell later how to proceed") - not yet wired into any
    SquareItem field, condition, or CoreItem; which n-ary terms should
    actually get rewritten to reference a compound id, and how a compound
    id's own eventual "chosen" status gets determined from its own
    constituents, is not yet decided.
    """
    ids = tuple(ids)
    entries = []
    counter = 0
    for size in range(2, len(ids) + 1):
        for members in combinations(ids, size):
            entries.append((("compound", counter), members))
            counter += 1

    compound_ids = set()
    for cid, members in entries:
        if len(members) <= 3:
            members_set = set(members)
            is_subset_of = frozenset(
                other_cid for other_cid, other_members in entries
                if other_cid != cid and members_set <= set(other_members))
        else:
            is_subset_of = frozenset()
        compound_ids.add((cid, members, is_subset_of))
    return compound_ids


def generate_compound_candidates(tile_choices):
    """Added 2026-11-01 - refines create_compound_ids's own generation step
    (Ivo, same day: "a subtle change in our generation of compound ids").

    create_compound_ids took a flat set of ids and enumerated every
    non-trivial subset, as if any id could independently coexist with any
    other. That's too permissive: in reality a tile's borders are reached
    by several DIFFERENT placement tiles (Ivo's own a/b/c), and a single
    placement tile can itself offer more than one alternative choice (b1
    or b2 - two different placements on tile b) - but only ONE of a given
    placement tile's own alternatives can ever be chosen at once, since
    they occupy the same placement tile. A flat subset-of-ids enumeration
    doesn't know this, and would (wrongly) allow e.g. {b1, b2} together.

    tile_choices is a mapping {tile: (choice_1, choice_2, ...)} - one entry
    per placement tile, its own value the tuple of ids it could contribute
    (a tile with only one way to reach the target cell has a single-id
    tuple, like Ivo's own a=(id1,)).

    Returns a set of (members, is_subset_of) pairs ("the sets" - Ivo added
    "is subset of" the same day, filled in here the same way
    create_compound_ids already fills it for its own entries), one per way
    to pick one or more placement TILEs and then exactly one choice from
    each picked tile: for every subset of tile_choices's own keys of size
    >= 1, the cross product of those tiles' own choice-tuples, one
    combination per member of the product. For >=2 tiles this is Ivo's
    own worked example exactly: tiles {a: (id1,), b: (id2, id3), c: (id4,
    id5)} produces the same 12 sets Ivo listed by hand ({a,b1}, {a,b2},
    {b1,c1}, ... {a,b2,c2}).

    The size->=2 floor from the first version of this function is GONE
    (Ivo, 2026-11-01: "drop the condition 'number of ids in a compound
    >1'... turn id1, id2 ... the single items into compounds and put them
    on the list") - a single tile's own single choices now also become
    their own size-1 candidates (id1, id2, id3, id4, id5 each get their
    own singleton entry too, on top of the 12 multi-tile ones above). A
    size-1 candidate still gets a real is_subset_of (every bigger
    candidate containing it, same <=3 cap as everything else) - it's only
    the floor on candidate SIZE that's gone, not the is_subset_of rule.
    create_compound_ids's own floor (size >= 2 of a flat id list, a
    different function for a different case - point 3 of the downstream
    plan) is untouched; this change is to generate_compound_candidates
    only.

    members is a frozenset - itself the candidate's own unique handle, no
    separate arbitrary compound id needed (unlike create_compound_ids,
    which needs one because its own entries get referenced BY id
    elsewhere; nothing here does yet). is_subset_of is the frozenset of
    every OTHER candidate's own members that is a proper superset of this
    entry's own members - computed only when len(members) <= 3, same rule
    and same reasoning as create_compound_ids's own is_subset_of ("only a
    size-<=3 term could ever substitute into a real board condition"),
    always the empty frozenset above that.

    Deliberately stops at the sets and is_subset_of - no compound id
    values assigned yet (unlike create_compound_ids); Ivo's own
    instruction this round was only "create functions to generate the
    sets, and to prune them. No use yet." - prune_compound_candidates and
    prune_blocked_compound_candidates (below) are the companion pruning
    steps; neither this nor they are wired into anything.
    """
    tiles = tuple(tile_choices)
    all_members = []
    for size in range(1, len(tiles) + 1):
        for subset in combinations(tiles, size):
            for combo in product(*(tile_choices[tile] for tile in subset)):
                all_members.append(frozenset(combo))

    candidates = set()
    for members in all_members:
        if len(members) <= 3:
            is_subset_of = frozenset(
                other for other in all_members
                if other != members and members <= other)
        else:
            is_subset_of = frozenset()
        candidates.add((members, is_subset_of))
    return candidates


def prune_compound_candidates(candidates, forces):
    """Added 2026-11-01, companion to generate_compound_candidates (same
    Ivo instruction: "create functions to generate the sets, and to prune
    them").

    generate_compound_candidates's own cross product doesn't know about
    forcing relationships BETWEEN different placement tiles' choices - e.g.
    Ivo's own "id2 (= b1) forces id4 (= c1)": choosing b1 necessarily also
    means id4 ends up chosen, whether or not tile c's own slot was even
    included in a given candidate, and REGARDLESS of which of tile c's own
    alternatives that candidate happened to name. So a candidate naming b1
    without also naming c1 (whether c is missing entirely, like {a, b1}, or
    present with the wrong choice, like {b1, c2}) describes something that
    can't actually happen - it has to be discarded.

    candidates is generate_compound_candidates's own (members, is_subset_of)
    pairs - this function only ever looks at each pair's own members, never
    its is_subset_of, and passes a surviving entry through unchanged
    (is_subset_of still describes that entry's relationship to the full,
    pre-pruning candidate universe, same as create_compound_ids's own
    is_subset_of is never pruning-aware either).

    forces is an iterable of (id_a, id_b) pairs, same direction as
    SquareItem.forces itself (id_a forces id_b - id_b only ever becomes
    eligible once id_a is chosen). Returns a new set containing only the
    (members, is_subset_of) pairs from `candidates` where, for every
    (id_a, id_b) in forces, id_a in members implies id_b in members too.

    Matches Ivo's own worked example exactly: with forces = {(id2, id4)},
    discards exactly {a, b1} (= {id1, id2}), {b1, c2} (= {id2, id5}), and
    {a, b1, c2} (= {id1, id2, id5}) out of the 12 candidates above, keeping
    the other 9 - "it discards {a,b1}, {b1,c2}, and {a,b1,c2}," Ivo's own
    words.
    """
    return {
        (members, is_subset_of) for members, is_subset_of in candidates
        if all(id_b in members for id_a, id_b in forces if id_a in members)
    }


def prune_blocked_compound_candidates(candidates, blocked):
    """Added 2026-11-01, second companion to generate_compound_candidates
    (Ivo, same day: "check the compound list also against the .blocked
    list. remove blocking items from the compound list").

    prune_compound_candidates (above) removes candidates contradicted by a
    FORCING relation. This is the same idea for `.blocked` instead:
    `.blocked` (the former cores.CoreItem.update_blocked, removed 2026-11-07) is a set of (id1, id2) pairs
    - unordered, same "stored low-id-first" convention as .conflicts -
    meaning id1 and id2 can never both be chosen, period, regardless of
    which placement tiles they came from. A compound candidate is a claim
    that every one of its own members IS chosen simultaneously, so a
    candidate naming both halves of a `.blocked` pair describes something
    that can never actually happen, exactly like a forcing contradiction -
    it has to be discarded too.

    candidates is generate_compound_candidates's own (members, is_subset_of)
    pairs (same convention as prune_compound_candidates above) - only
    members is inspected, is_subset_of passes through unchanged on a
    surviving entry.

    blocked is an iterable of (id1, id2) pairs (order within a pair doesn't
    matter - checked both ways). Returns a new set containing only the
    (members, is_subset_of) pairs from `candidates` whose members contain
    at most one id from every blocked pair.

    Worked extension of Ivo's own example: with the same 9 candidates
    prune_compound_candidates already kept (forces = {(id2, id4)}), adding
    blocked = {(id3, id4)} (b2 and c1 mutually exclusive) discards {b2, c1}
    (= {id3, id4}) and {a, b2, c1} (= {id1, id3, id4}) too, leaving 7.
    """
    return {
        (members, is_subset_of) for members, is_subset_of in candidates
        if not any(id1 in members and id2 in members for id1, id2 in blocked)
    }


def get_or_create_compound(compounds, members):
    """Added 2026-11-01, shared helper for compoundize_forced_by/
    propagate_compound_links (below). `compounds` is a registry dict
    `{members_frozenset: compound_id}`, built up incrementally (unlike
    create_compound_ids/generate_compound_candidates, which both enumerate
    a whole pool up front) - "if possible" (Ivo, 2026-11-01: "replace ...
    by compounds if possible") means "if a matching compound already
    exists in the registry," with this function as the fallback that
    mints one on first use rather than requiring it pre-built. Same
    arbitrary-but-unique, tagged convention as create_compound_ids's own
    compound ids (`("compound", n)`, never colliding with a real (row,
    col) cell id) - the counter is just `len(compounds)`, safe because
    entries are only ever added, never removed.

    Returns the existing compound id for `members` if already registered
    (mutates nothing), otherwise creates, registers, and returns a fresh
    one.
    """
    members = frozenset(members)
    if members not in compounds:
        compounds[members] = ("compound", len(compounds))
    return compounds[members]


def compoundize_forced_by(m, realm, compounds=None):
    """Added 2026-11-01 (Ivo: "now replace all ids, tuple terms and triple
    terms in .forced_by and .forces in the 3x3 of SquareItems by compounds
    if possible"). Step 1 of 2 - see propagate_compound_links for step 2.

    For every cell in `realm` (same (( row_start, row_end), (col_start,
    col_end)) convention as forced_by_on_realm/compute_eligibility), for
    every term in that cell's own `.forced_by`:
    - if a compound already exists (in `compounds`, the registry - see
      get_or_create_compound) whose own members are EXACTLY this term's
      full id set, the whole term is replaced by that one compound - "if
      possible" succeeds, collapsing a tuple/triple term straight to a
      single compound reference in one step.
    - otherwise ("if possible" fails - no compound for the FULL term
      exists yet), each individual id in the term is replaced by ITS OWN
      singleton compound instead (minted on first use, reused on every
      later occurrence of that same id anywhere in `realm`) - the term's
      own arity is preserved (a triple of ids becomes a triple of
      singleton compounds, not collapsed), since there is nothing yet to
      collapse it to. This is always possible, so a single bare id (a
      size-1 term, e.g. after collapse_synced_ids's own dedup) always
      ends up wrapped in its own singleton compound too - "turn id1, id2
      ... the single items into compounds" (Ivo, same instruction as
      generate_compound_candidates's own size->=1 change).

    Every compound-tagged entry written here uses PlacementStatus.unknown,
    uniformly, regardless of what the original real id's own tag said -
    deliberately NOT carrying over a real id's own placement_status,
    since a compound is a different entity whose own "chosen" status
    (are ALL of its members chosen?) is a standing open design question
    (see TODO.md item 2's own note: "whether a virtual id can itself
    become a member of some FURTHER condition... Recommend deciding up
    front") - asserting a borrowed per-id status here would misrepresent
    that, not yet-decided, fact.

    `.forces` is cleared (`item.forces = set()`), not rebuilt the normal
    way (bruteforce.update_forces): once every term's ids are replaced by
    compounds, no REAL id appears as a forcer anywhere in `realm` any
    more, so a real cell's own `.forces` (keyed by real, array-indexable
    ids) has nothing left to hold - update_forces's own `m[a].forces = ..`
    line would in fact raise, since a compound id isn't a valid array
    index. Where this information actually goes now is the separate
    `compound_forces` registry propagate_compound_links builds - "a
    forced_by list for compounds to terminals" (Ivo) lives there, not on
    any SquareItem.

    Mutates every cell's `.forced_by`/`.forces` in `realm`, in place.
    `compounds` is also mutated in place (new entries added via
    get_or_create_compound) and returned, so the same registry can be
    threaded into propagate_compound_links afterward; pass `None` (the
    default) to start a fresh one.
    """
    if compounds is None:
        compounds = {}
    (row_start, row_end), (col_start, col_end) = realm
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            item = m[i, j]
            new_forced_by = set()
            for term in item.forced_by:
                ids_in_term = frozenset(id_ for id_, _status in term)
                if ids_in_term in compounds:
                    cid = compounds[ids_in_term]
                    new_forced_by.add(frozenset({(cid, PlacementStatus.unknown)}))
                else:
                    new_forced_by.add(frozenset(
                        (get_or_create_compound(compounds, frozenset({id_})), PlacementStatus.unknown)
                        for id_ in ids_in_term))
            item.forced_by = new_forced_by
            item.forces = set()
    return compounds


def propagate_compound_links(m, realm, compounds):
    """Added 2026-11-01, step 2 of 2 - see compoundize_forced_by (step 1)
    for the "replace by compounds if possible" half this continues from.
    Must be called after compoundize_forced_by has already converted
    every term in `realm` to compounds (same `m`/`realm`/`compounds` as
    that call).

    Ivo's own worked illustration (paraphrased): a term (A, B, C) in some
    cell id6's own `.forced_by`, with A/B/C all compounds (exactly
    compoundize_forced_by's own "preserve arity" fallback case above) -
    "find the compound Q that is the union of A, B and C. The such-created
    new link is only for Q." Then: "a link for A is also a link for any
    compound id P that A is subset of ... all links for A,B,C that lead to
    the Q link above are also links for Q" - i.e. Q, being a strict
    superset of A/B/C's own members, inherits everything each of them
    forces too, since choosing Q's own members necessarily means A's (and
    B's, and C's) own members all got chosen along with it.

    Two passes over every cell b in `realm`:

    1. Build compound_forces - the new "forced_by list for compounds to
       terminals" (Ivo) this function returns - a dict {compound_id: set
       of real ids}, the compound-side analog of SquareItem.forces, kept
       in a separate dict rather than on any SquareItem for the same
       reason compoundize_forced_by clears `.forces` (no array slot for a
       compound id). For every term in b's own (already-compoundized)
       `.forced_by`, every compound named in it gets b added to its own
       compound_forces entry - same "participates in" semantics as the
       real .forces field (SquareItem.forces's own docstring: B gets
       added to A's forces whenever A appears in any of B's own .forced_by
       conditions, not only when A alone, unconditionally, forces B) - so
       a direct size-1 term {(A, _)} means "A forces id7" outright, while
       A merely appearing alongside B/C in a size-3 term means "A
       participates in forcing id6," exactly mirroring the real field's
       own looser relationship.

    2. For every multi-compound term (A, B, C, ...) found in some cell
       b's own `.forced_by` (size > 1 - a size<=1 term has nothing to
       union, already fully handled by pass 1): compute `full_members`,
       the union of every one of the term's own compounds' own members;
       `Q = get_or_create_compound(compounds, full_members)` finds Q if
       some earlier call already created it (for this exact combination,
       possibly discovered via a totally different cell's own term), or
       mints it fresh otherwise. Adds b directly to Q's own
       compound_forces entry ("the such-created new link is only for Q" -
       not for A, B, or C individually at this new link, though each of
       them may separately already force other things of their own from
       pass 1). Also records this new fact back into b's own `.forced_by`
       itself, as a new size-1 term {(Q, unknown)} - .forced_by and the
       compound_forces side table are kept as two views of the same fact,
       same discipline as the real .forced_by/.forces pair.

       Then inherits: for EVERY other compound in the registry (not just
       A, B, and C - "and others", Ivo's own words) whose own members are
       a subset of `full_members`, Q's own compound_forces entry absorbs
       that compound's own compound_forces entry too. This is not capped
       at <=3 members the way create_compound_ids/generate_compound_
       candidates's own is_subset_of is - that cap was about whether a
       SMALL compound bothers recording a pointer up to its own bigger
       supersets (only relevant once it could actually substitute into a
       real, <=3-member board condition); this is the reverse query, FROM
       Q looking down at whatever already exists beneath it, with no such
       cap.

    Mutates every b's own `.forced_by` in `realm` (adding Q links) and
    `compounds` (new Q entries) in place. Returns compound_forces.
    """
    members_by_compound = {cid: members for members, cid in compounds.items()}
    (row_start, row_end), (col_start, col_end) = realm

    compound_forces = {}
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            b = (i, j)
            for term in m[b].forced_by:
                for cid, _status in term:
                    compound_forces.setdefault(cid, set()).add(b)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            b = (i, j)
            for term in list(m[b].forced_by):
                if len(term) <= 1:
                    continue
                cids = {cid for cid, _status in term}
                full_members = frozenset().union(*(members_by_compound[cid] for cid in cids))
                q = get_or_create_compound(compounds, full_members)
                members_by_compound[q] = full_members
                compound_forces.setdefault(q, set()).add(b)
                for other_cid, other_members in members_by_compound.items():
                    if other_cid != q and other_members <= full_members:
                        compound_forces[q] |= compound_forces.get(other_cid, set())
                m[b].forced_by.add(frozenset({(q, PlacementStatus.unknown)}))
    return compound_forces


def compute_eligibility(m, realm):
    """Set every cell's .placement_status relative to realm, in place, then replace
    every inside cell's own .forced_by with the fully flattened n-ary terms
    that actually explain it - not the original 3-element conditions, which
    may still name an inside cell rather than what makes that cell eligible.

    This replaces the earlier, geometric in_realm-only carry-up rule used in
    the former cores.CoreItem.update_dependencies, removed 2026-11-07 (see squareplacement_superlattice_
    refactor memory note - that rule was wrong): a condition fully inside
    realm is not automatically trustworthy just because every member happens
    to be geographically inside - each of those members can itself only
    become chosen via its own chain of conditions, and that chain can in turn
    depend on cells further outside realm that this core knows nothing about.
    Geography alone cannot settle that; this function actually traces it.

    A cell outside realm is set placement_status=PlacementStatus.eligible
    unconditionally - this core tracks nothing about it, so there is no basis
    to rule it out, and treating it optimistically as possible is what lets
    the trace below terminate at all (see squareplacement_superlattice_refactor
    memory note, "the forcing recursion never terminates without an active
    placement" - an outside cell defaulting to eligible is the boundary
    condition that grounds it, standing in for "something outside this view
    may actively place this").

    Every cell inside realm starts placement_status=PlacementStatus.unknown
    and justified (its own set of fully flattened terms, see below) empty. A condition
    (a, b, c) of some inside cell X only ever contributes a term once every
    one of a, b, c is itself eligible - but contributes more than one term,
    and more than a plain 3-element one, once it does: each member m is
    replaced by every one of m's own justified terms if m is itself inside
    realm (duplicating the whole combination once per such term - a member
    inside realm can be justified more than one way, and each way needs its
    own flattened term, not just one arbitrarily picked), or just {m} itself
    if m is outside realm (nothing to substitute, it's already a ground
    fact). Every combination across (a, b, c)'s own substitution options is
    unioned into one flat set - "flatten the term to an n-ary": the result
    names only ids that are themselves either outside realm or recursively
    explained the same way, never an inside id whose own explanation is
    still one hop further back, and it can be wider than 3 members once any
    substitution actually happened. A flattened term is discarded outright,
    never added to X's own justified terms, if _has_diagonal_conflict finds
    two of its own members diagonal to each other - vacuum_condition_offsets
    only ever excluded that within one original triple, and flattening
    combines members from different triples, so nothing else catches a pair
    that only becomes diagonal once combined (confirmed this is a real,
    not hypothetical, gap - about a third of one small worked example's own
    flattened terms were self-blocking this way before this check existed).
    Iterated to a fixed point (a full pass adds no new term to any cell's
    justified set) - lets a cell's own justified terms keep growing as the
    cells it depends on keep discovering more of their own. X.placement_status
    is set to PlacementStatus.eligible as soon as justified[X] is nonempty;
    nothing here ever sets it to unreachable, only eligible or left at
    unknown - see SquareItem.placement_status's own docstring.

    Once that converges, every inside cell's own .forced_by is replaced
    outright by its own justified set - not filtered from the original
    conditions, which no longer appear at all once their every member has
    been substituted away. A cell whose justified set stayed empty - no
    condition of its own was ever fully traceable to eligible ids - ends up
    with .forced_by = set(), the same as if it had none to begin with; a
    bigger realm can always reconstruct more from scratch by running this
    same function over its own, larger realm.

    Mutates .forced_by, so update_forces(m) is called at the end to keep
    .forces consistent with it - see that function's own docstring.
    """
    rows, cols = m.shape
    inside = [(i, j) for i in range(rows) for j in range(cols)
              if in_realm((i, j), realm)]
    inside_set = set(inside)
    for i in range(rows):
        for j in range(cols):
            m[i, j].placement_status = (PlacementStatus.eligible if (i, j) not in inside_set
                                         else PlacementStatus.unknown)

    justified = {cell: set() for cell in inside}

    def term_options(member):
        # member is a (id, placement_status) pair, not a plain id - see
        # SquareItem.forced_by's own docstring; justified is keyed by plain
        # id, so look up by member's id half, but still substitute the
        # tagged pair itself (not the bare id) when member is a ground fact.
        member_id, _member_status = member
        return justified[member_id] if member_id in inside_set else {frozenset({member})}

    changed = True
    while changed:
        changed = False
        for cell in inside:
            item = m[cell]
            before = len(justified[cell])
            for condition in item.forced_by:
                for combo in product(*(term_options(member) for member in condition)):
                    term = frozenset().union(*combo)
                    if not _has_diagonal_conflict(term):
                        justified[cell].add(term)
            if len(justified[cell]) > before:
                changed = True
            if justified[cell]:
                item.placement_status = PlacementStatus.eligible

    for cell in inside:
        m[cell].forced_by = justified[cell]
    update_forces(m)
