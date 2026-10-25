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


def compute_eligibility(m, realm):
    """Set every cell's .placement_status relative to realm, in place, then replace
    every inside cell's own .forced_by with the fully flattened n-ary terms
    that actually explain it - not the original 3-element conditions, which
    may still name an inside cell rather than what makes that cell eligible.

    This replaces the earlier, geometric in_realm-only carry-up rule used in
    cores.CoreItem.update_dependencies (see squareplacement_superlattice_
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
