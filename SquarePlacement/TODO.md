# TODO

Open design/implementation items for the condition-based closure redesign
(conditions.py/bruteforce.py/cores.py). See also OPTIMIZATIONS.md for
performance-specific candidates - this file is for correctness/design work,
not efficiency.

## 0. `.forces`: interim supersupertile scope, not a precise margin-3

`conditions.forced_by_on_realm` (added 2026-10-31) computes `.forced_by`
for just one core's own cells, without needing the rest of the board -
"compute on the superlattice," confirmed safe because FORCED_BY_REACH (2
cells) bounds every real triple member, and needs no actual neighbour
data - just the true board shape (`init_vacuum` is pure offset
arithmetic). `.forces` was deliberately left out of that function (Ivo,
2026-10-31: "only switch `.forced_by` to per-core computation now").

The two fields aren't actually symmetric here: `.forces`'s *initial*
build (inside `conditions.initial_map`, via `init_vacuum`'s own cached
`_forces_offsets`) is just as purely geometric as `.forced_by` is - no
margin issue there either. The real margin-3 requirement only shows up
for `bruteforce.update_forces`'s own ongoing maintenance, which rebuilds
`.forces` by inverting every nearby cell's CURRENT (real,
placement-mutated) `.forced_by` - genuinely needs real neighbour data
within reach 3, not just the true board shape, every time `.forced_by`
changes (`compute_conditions`, `prune_same_core_candidates`,
`retag_forced_by` all call it).

Decided 2026-10-31 (interim, not a precise solution): when `.forces`
needs refreshing for a core, run `bruteforce.update_forces` scoped to
that core's own level-2 CoreItem realm (a "supersupertile" - 3x3 of 3x3
of base tiles) rather than the whole board, or a precise margin-3 window.
Comfortably correct for any cell away from the supersupertile's own outer
edge; wasteful (recomputes far more than the minimal margin needs) by
design - tracked as OPTIMIZATIONS.md item 6, to revisit with a real
`forces_on_realm`-style margin-3 window once the precision is worth
building.

## 1. RESOLVED (2026-10-31): doubly-external `.forced_by` pairs during composition

Found 2026-10-31: `_compose_up`'s own union can carry a `.forced_by` pair
into a composed parent even when BOTH of its ids are external to every one
of the parent's own children (confirmed concretely - `((6,9),(9,6))`
already existed verbatim in child (2,2)'s own `.forced_by`, a genuine
"chain passes all the way through tile (2,2)" fact, carried up unchanged
even though neither endpoint is inside the bigger parent either).

Resolved: such a pair is never actually ownerless, and this is structural,
not merely unobserved. `_compose_up`'s own BFS (see its docstring) only
records a result the moment a hop already lands outside the realm being
composed - so a doubly-external pair's raw `(id_in, id_out)` edge must
already have existed, unchanged, as a literal member of exactly one
child's own `.forced_by` before this core's union ever ran. And it stays
true recursively: anything `push_down` itself ever pushes into a child's
own `.forced_by` always has an `id_in` that is a confirmed
`PlacementStatus.candidate` (the only way an interior `id_in` gets into
`.forced_by` at all - see `update_forced_by`'s own docstring), i.e. one
fixed physical cell, which is interior to exactly one grandchild too - so
`owning_child` always succeeds again one level further down. There is no
path, upstream or downstream, that ever produces a `.forced_by` pair
genuinely homeless in every child.

Given that, `push_down` needs no special-case code for this at all - it's
back to simply skipping an entry when `owning_child(id_in)` finds no
match (same as its very first version), now correctly understood as "it's
already exactly where it belongs" rather than "nothing to do because
nothing can be done." An `already_homed`/assert check was added briefly
to verify this before the structural argument was nailed down, then
removed again once confirmed unreachable (2026-10-31). Test:
`test_push_down_leaves_a_doubly_external_forced_by_pair_where_it_already_is`
(test_context_tiles.py).

## 2. Fill in the two downstream stubs

`bruteforce.collapse_synced_ids(m, id0, synced_ids)` and
`bruteforce.create_virtual_sync_id(m, ids)` (both added 2026-10-31) are
currently stubs that raise `NotImplementedError` - see their own docstrings
for the full intended design (Ivo's own 3-point plan, same date). Needs a
concrete worked example first, same methodology as everything else built
this session - not designed up front.

## 3. Recompute ordering and convergence across the CoreItem pyramid

We now have several `update_*`/`push_down` methods that read and write each
other's output, and it's not yet formalized what order they need to run in,
or whether one pass ever suffices. Traced directly from the code
(2026-10-31):

- **Within one level, going up**: `update_placement_reach()` must run
  before `update_forced_by()` (the recursive branch reads
  `self.placement_reach` directly - see its own docstring). `update_blocked()`
  needs the children's own `.forced_by`/`.placement_reach`/`.blocked` all
  already complete, which means the children themselves must have already
  finished their own `update_placement_reach` -> `update_forced_by` ->
  `update_blocked` sequence before a parent can even start its own.
- **The "down" passes mutate what the "up" pass already read**:
  `update_aliases_down`'s own consumption step (`_consume_aliases_down`)
  and `push_down` both write directly into a child's own `.forced_by`/
  `.blocked` - fields the parent's own `update_forced_by`/`update_blocked`
  already read to produce ITS OWN result. So a parent's own already-computed
  `.forced_by`/`.blocked` can go stale the moment a down-pass changes what
  it was built from - potentially revealing new composed facts that weren't
  visible the first time up.
- **Open**: does one full up-pass (base to top) followed by one full
  down-pass (top to base) actually reach a fixed point, or can a down-pass
  always reveal something a repeated up-pass would still need to pick up
  (requiring iteration: up, down, up, down, ... until nothing changes)?
  Not yet proven either way. The whole process is still a monotone closure
  (every mutation only ever ADDS entries to `.forced_by`/`.blocked`, never
  removes them - same argument already confirmed 2026-10-03 for the
  broader design), so it's bounded and must eventually terminate - but
  "terminates" isn't the same as "terminates in one pass," and that's the
  open part.

## 4. Possible explosion in `create_virtual_sync_id` - assessed, likely NOT
   a real issue, but worth confirming once implemented

Raised 2026-10-31: if many different placement tiles all contribute ids
that are mutually "syncable" (point 3 of the downstream plan - see #2
above), does enumerating every non-trivial subset (2^n - n - 1 for n ids)
blow up combinatorially, the way `compute_eligibility`'s own
flatten-and-substitute approach did (2026-10-14/15)?

Preliminary analysis (not yet verified against a real implementation,
since the function is still a stub): likely bounded, NOT exponential in
the total number of placement tiles on the board - `create_virtual_sync_id`
should only ever be scoped to the members of ONE shared vacuum condition at
a time, and every vacuum condition is capped at 3 members (pre-reduction).
For 3 ids that's exactly 4 subsets (3 pairs + 1 triple, matching Ivo's own
worked example), not a function of how many placement tiles exist overall.
The real risk would only appear if this were (incorrectly) called with an
unscoped set of "all syncable candidates anywhere," which isn't the
intended usage.

One thing that DOES need an explicit decision once this is implemented:
whether a virtual id can itself become a member of some FURTHER condition,
recursively (if so, that reintroduces something close to
compute_eligibility's own disaster - flattening-with-substitution across
multiple levels). Recommend deciding up front that virtual ids do NOT
recurse into further virtual-id creation, rather than discovering this the
hard way.
