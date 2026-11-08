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

## 2. Downstream stubs - step 1 and 2 implemented, step 2's "how to proceed" still open

**Tests for `create_compound_ids`, `generate_compound_candidates`,
`prune_compound_candidates`, and `prune_blocked_compound_candidates` now
live in `test_compounds.py`, not `test_context_tiles.py`** (moved
2026-11-01, Ivo: "make your tests explicit and put them into a file
test_compounds.py") - also rewritten to assert the full expected literal
data directly rather than deriving/checking it piecemeal; the two
`generate_compound_candidates` tests referenced by name below (`..
matches_ivos_own_worked_example` / `..fills_is_subset_of_for_small_terms`)
were merged into one explicit test there. `collapse_synced_ids`'s own
tests stayed in `test_context_tiles.py` - only the compound-id functions
moved.

Ivo's own base-level ordering (2026-11-01):

1. **`bruteforce.collapse_synced_ids(m, id0, synced_ids)` - implemented.**
   Replaces every occurrence of every id in `synced_ids` with `id0` across
   every SquareItem's own `.forced_by`, map-wide, re-tagging each
   replacement's embedded `placement_status` to `id0`'s current one; a
   condition naming two or more of `synced_ids` collapses for free via
   frozenset dedup ("creating more links"). Calls `update_forces(m)` at
   the end to keep `.forces` consistent. Deliberately does NOT recompute
   the `CoreItem` pyramid ("don't recompute the tile yet") - that stays
   the caller's job, same discipline as `unblock_path`. The dedup carries
   all the way down: if every member of a condition maps to `id0` (a pair
   `{id1, id2}` both syncing), it collapses to a plain size-1 condition
   `{id0}` - "(id0, id0) becomes id0" - with no special case needed, same
   frozenset mechanism as the partial-collapse case. Tests:
   `test_collapse_synced_ids_simplifies_a_triple_and_updates_forces`,
   `test_collapse_synced_ids_collapses_a_pair_entirely_into_a_single_link`.

2. **`bruteforce.create_compound_ids(ids)` - implemented, renamed from
   `create_virtual_sync_id` ("virtual id" -> "compound id").** Builds and
   returns `compound_ids`: a set of `(compound_id, members, is_subset_of)`
   triples (expanded from `(compound_id, members)` pairs, Ivo, 2026-11-01),
   one per non-trivial subset (size >= 2, including n > 3 - "I still
   maintain n-aries with n>3 matter") of `ids`. A compound id's own value
   is arbitrary (just unique, tagged `("compound", n)` so it can never
   collide with a real `(row, col)` id) rather than the earlier plan's
   deterministic `frozenset(members)` encoding - the mapping lives in the
   returned set instead. `is_subset_of` names every OTHER compound id
   whose own `members` is a superset of this entry's own - computed only
   when `len(members) <= 3` ("Do this only for A with number of id1,id2
   ... <=3. Empty otherwise" - Ivo's own words, matching the standing
   "a real board condition is capped at 3 members" fact), always the
   empty `frozenset` above that, even when a real superset exists. Tests:
   `test_create_compound_ids_enumerates_every_subset_size_two_and_up`,
   `test_create_compound_ids_is_subset_of_links_small_terms_to_their_supersets`.
   **Deliberately stops here** - not wired into any `SquareItem` field,
   condition, or `CoreItem` yet; Ivo said twice "I will tell later how to
   proceed," so the next step (which n-ary terms get rewritten to
   reference a compound id, and how a compound id's own "chosen" status
   gets determined) is still open and intentionally unbuilt.

   **Refined 2026-11-01 - a subtle but real correction to the generation
   step (Ivo, same day: "a subtle change in our generation of compound
   ids").** `create_compound_ids`'s flat-subset-of-`ids` enumeration is too
   permissive: in the real board, the ids reaching one tile's border don't
   float free - each comes from some specific placement tile, and a
   placement tile that offers more than one way to reach the border
   (Ivo's own b1/b2 on tile "b") offers MUTUALLY EXCLUSIVE alternatives,
   since they're different placements on the very same spot - a flat
   subset enumeration doesn't know this and would wrongly allow e.g.
   `{b1, b2}` together. The correct combinatorics (Ivo's own worked
   example, verbatim): "Do combinatorics of tiles: all subsets of
   `{a,b,c}` with more than one member... then there are two choices on
   tile b and two on tile c" - i.e. enumerate subsets of PLACEMENT TILES
   (size >= 2), then cross-product one choice per tile in that subset.
   With `a=(id1,)`, `b=(id2,id3)`, `c=(id4,id5)`, this is exactly the 12
   sets Ivo listed by hand (`{a,b1}, {a,b2}, {b1,c1}, ... {a,b2,c2}`).

   On top of that, a further complication: one id can FORCE another id
   that belongs to a different placement tile (Ivo's own "id2=b1 forces
   id4=c1") - meaning a candidate naming `b1` is only real if it also
   names `c1`, whether tile c was left out of the candidate entirely
   (`{a,b1}`) or included with the wrong alternative (`{b1,c2}`); such
   candidates have to be pruned.

   **Implemented, "no use yet" per Ivo's own instruction** (exact words:
   "Create functions to generate the sets, and to prune them. No use
   yet."):
   - `bruteforce.generate_compound_candidates(tile_choices)` - takes
     `{tile: (choice_1, choice_2, ...)}`, returns the set of frozensets
     from the tile-subset x per-tile-choice cross product described
     above. Verified against Ivo's own 12-set worked example. Test:
     `test_generate_compound_candidates_matches_ivos_own_worked_example`.
   - `bruteforce.prune_compound_candidates(candidates, forces)` - takes
     that set plus an iterable of `(id_a, id_b)` forcing pairs (same
     direction as `SquareItem.forces`), returns only the candidates where
     every `id_a` present also has its forced `id_b` present. Verified
     against Ivo's own worked example: with `forces = {(id2, id4)}`,
     discards exactly `{a,b1}`, `{b1,c2}`, `{a,b1,c2}`, keeps the other 9
     - Ivo's own words, matched exactly. Test:
     `test_prune_compound_candidates_discards_candidates_contradicting_a_forcing_relation`.

   Neither function assigns compound id values or an `is_subset_of` yet
   (unlike `create_compound_ids`) and neither is wired into anything -
   this refines the GENERATION step only; "how to proceed" from here is
   still Ivo's to decide.

   **Third pruning pass added (2026-11-01, same day, Ivo: "check the
   compound list also against the .blocked list. remove blocking items
   from the compound list").** `prune_compound_candidates` only checks
   forcing; `.blocked` (`cores.CoreItem.update_blocked`) is a separate,
   unordered set of `(id1, id2)` pairs meaning id1 and id2 can never both
   be chosen, regardless of which placement tile either came from - a
   compound candidate naming both halves of a blocked pair is just as
   impossible as one contradicting a forcing relation, so it needs the
   same kind of discard.
   - `bruteforce.prune_blocked_compound_candidates(candidates, blocked)` -
     drops any candidate containing both ids of any `(id1, id2)` in
     `blocked`. Verified by extending Ivo's own worked example: starting
     from the 9 candidates `prune_compound_candidates` already kept
     (`forces={(id2,id4)}`), adding `blocked={(id3,id4)}` (b2/c1 mutually
     exclusive) discards `{b2,c1}` and `{a,b2,c1}`, leaving 7. Test:
     `test_prune_blocked_compound_candidates_discards_candidates_naming_both_halves_of_a_blocked_pair`.
   Still "no use yet" - not wired into anything.

   **`is_subset_of` filled in for `generate_compound_candidates` too
   (2026-11-01, same day, Ivo: "also fill the "is subset of" - field of
   the compounds, if not already done").** It wasn't done yet -
   `generate_compound_candidates` now returns a set of `(members,
   is_subset_of)` pairs instead of bare frozensets, same rule as
   `create_compound_ids`'s own `is_subset_of`: every OTHER candidate's own
   `members` that is a proper superset of this entry's, computed only
   when `len(members) <= 3` (every entry qualifies in the running example,
   since the tile-grouped generation never exceeds 3 picked tiles there).
   No separate arbitrary compound id was introduced for these entries -
   `members` (a frozenset) already serves as its own unique handle, unlike
   `create_compound_ids`, which needs a real id because its own entries
   get referenced BY id elsewhere. `prune_compound_candidates` and
   `prune_blocked_compound_candidates` both updated to operate on
   `(members, is_subset_of)` pairs - only `members` is inspected by either,
   and a surviving entry's `is_subset_of` passes through unchanged
   (describes the entry's relationship to the full, pre-pruning candidate
   universe, same as `create_compound_ids`'s own `is_subset_of` is never
   pruning-aware either). Tests:
   `test_generate_compound_candidates_fills_is_subset_of_for_small_terms`,
   plus the three existing tests updated to unpack the new tuple shape.

   **Size->=2 floor dropped (2026-11-01, same day, Ivo: "drop the
   condition 'number of ids in a compound >1'. In other words, turn id1,
   id2 ... the single items into compounds and put them on the list").**
   `generate_compound_candidates`'s tile-subset loop now starts at size 1,
   not 2 - every individual choice from every tile (id1 through id5 in
   the running example) now gets its OWN singleton candidate too, on top
   of the 12 multi-tile ones (17 total). A singleton still gets a real
   `is_subset_of` (every bigger candidate containing it, same `<=3` cap as
   before) - only the floor on candidate SIZE was dropped, not the
   `is_subset_of` rule. Both pruning functions needed no code change, and
   behave correctly on singletons for the right reason: a singleton like
   `{id2}` is now correctly discarded by `prune_compound_candidates` when
   `forces` names `id2`, since "id2 alone" is impossible once id2 forces
   id4 (id4 always comes with it) - 4 discarded instead of 3 in the
   running example. A singleton can never be discarded by
   `prune_blocked_compound_candidates` on an ordinary 2-id blocked pair
   (it only names one id, never both halves), UNLESS the blocked pair is
   itself self-blocking (`(id0, id0)`, a real case `update_blocked`'s own
   docstring documents) - then the singleton correctly gets removed too,
   since id0 could never validly be chosen at all. `create_compound_ids`'s
   own size->=2 floor (flat id list, a different function/case) is
   untouched. Tests updated to include the 5 singletons and their
   is_subset_of/pruning behavior; full suite 39/39.

   **First actual wiring: `bruteforce.compoundize_forced_by`/
   `propagate_compound_links` (2026-11-01, same day) - Ivo's own "how to
   proceed" instruction, the thing twice deferred above.** Operates on a
   realm of real SquareItems (same `((row_start,row_end),(col_start,
   col_end))` convention as `forced_by_on_realm`/`compute_eligibility`),
   not on a pre-enumerated candidate pool - compounds are minted lazily,
   on demand, via a new shared helper:
   - `bruteforce.get_or_create_compound(compounds, members)` - `compounds`
     is a registry dict `{members_frozenset: compound_id}`, built up
     incrementally (unlike `create_compound_ids`/
     `generate_compound_candidates`, which both enumerate a whole pool up
     front). Same arbitrary-but-unique `("compound", n)` tagging.
   - `bruteforce.compoundize_forced_by(m, realm, compounds=None)` - step 1
     ("now replace all ids, tuple terms and triple terms in .forced_by
     and .forces in the 3x3 of SquareItems by compounds if possible" -
     Ivo). For every term in every realm cell's own `.forced_by`: if a
     compound ALREADY exists (in the registry) for the term's EXACT full
     id-set, the whole term collapses to that one compound ("if possible"
     succeeds). Otherwise, each individual id in the term is wrapped in
     ITS OWN singleton compound instead (minted on first use, reused on
     every later occurrence of that id), preserving the term's own arity
     - a triple of ids becomes a triple of singleton compounds, not
     collapsed, since nothing bigger exists for it yet. Every
     compound-tagged entry gets `PlacementStatus.unknown` uniformly - a
     compound's own "chosen" status (are ALL its members chosen?) is a
     standing open design question (see the "virtual id can itself
     become a member of some FURTHER condition" note at the end of #4
     below), so borrowing a real id's own status here would misrepresent
     that. `.forces` is cleared, not rebuilt the normal way
     (`bruteforce.update_forces` would crash - `m[a].forces = ..` isn't
     valid once `a` is a compound id, not an array index) - where this
     information goes now is the separate registry the next function
     builds.
   - `bruteforce.propagate_compound_links(m, realm, compounds)` - step 2,
     must run after step 1 on the same `m`/`realm`/`compounds`. Builds and
     returns `compound_forces`, a dict `{compound_id: set of real ids}` -
     "a forced_by list for compounds to terminals" (Ivo), the compound-
     side analog of `SquareItem.forces`, kept separately since a compound
     has no array slot of its own. Pass 1: same "participates in"
     semantics as the real `.forces` field - every compound named in any
     term of any realm cell b's own (already-compoundized) `.forced_by`
     gets b added to its own `compound_forces` entry. Pass 2 (Ivo's own
     worked illustration: "for a term (A,B,C) in the forced_by of some
     id6, with A, B, C compounds, find the compound Q that is the union
     of A,B and C. The such-created new link is only for Q"): for every
     remaining multi-compound term, unions every one of its own
     compounds' own members, gets-or-creates Q for that union, credits
     b to Q's own `compound_forces` entry directly, records a NEW size-1
     `{(Q, unknown)}` term back into b's own `.forced_by` (alongside the
     original multi-compound term, not replacing it), THEN inherits: "a
     link for A is also a link for any compound id P that A is subset
     of... all links for A,B,C that lead to the Q link above are also
     links for Q" - scans every OTHER compound in the registry ("and
     others", not just A/B/C) whose own members are a subset of Q's, and
     absorbs ITS `compound_forces` entry into Q's too (no `<=3`-members
     cap here, unlike `is_subset_of` elsewhere - that cap was about
     whether a small compound bothers pointing up to its own supersets;
     this is the reverse query, from Q looking down, with no such
     restriction). Verified against Ivo's own worked example directly (A
     forces id7; B forces id8; C forces id8; Q = union(A,B,C) ends up
     forcing exactly {id6, id7, id8}, not just the directly-created
     {id6}). Tests:
     `test_compoundize_forced_by_replaces_every_term_with_singleton_compounds`,
     `test_propagate_compound_links_creates_union_compound_and_inherits_forces`
     (test_compounds.py). Full suite 45/45 (verify against the latest run
     before trusting this count - confirm via `python -m pytest -q`).

   Still not wired into `compute_conditions`/the `CoreItem` pyramid/any
   actual caller outside these two tests - this is the mechanism now
   built and proven correct on a hand-built example, not yet hooked into
   the real per-round closure loop.

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

## 3a. "Last cell inside" convention (implemented 2026-11-07) - open downstream points

**Restructured 2026-11-08 towards CPU/GPU** (Ivo: "move the map, the
map_after and the .pyramid list out of the classes ... to a new class
DeviceMemory ... Base, Upstream and Downstream function will later become
kernel calls ... each step uses only data from one level"):
`DeviceMemory` holds all data between steps - `.map`, `.pyramid` (index =
level), `.inbox` (down-pass messages, laid out like `.pyramid`; was a local
variable between levels), `.map_after` (was `Base.map_onepass`), and the
constants `tile_size`, `origin`. `CoreItem` is data only. `Base`,
`Upstream`, `Downstream` hold no data; each has one level step plus the
per-CoreItem functions it applies:
`Base.build_level0(mem)` (.map -> .pyramid[0]),
`Upstream.build_level(mem, k)` (.pyramid[k-1] -> .pyramid[k]),
`Downstream.down_level(mem, k)` (.pyramid[k] + .inbox[k] -> .pyramid[k-1]
links + .inbox[k-1]; at k == 1 also .map_after). `Pyramid` is the CPU main
(`upward`, `downward`, `onepass(mem)`) plus the shared static helpers
(`in_realm`, `is_candidate`, `set_placement_tiles`, `owning_child`,
`child_exits`, `candidate_links`, `fold_into`, `empty_inbox`).
**Same day, preprocessing moved to Base** (Ivo: "The main (CPU) sets only
map.state ... Then Base calls some init function setting .condition etc"):
`Base.init_map(mem)` derives `.forced_by`, `.forces`, `.placement_status`
from `.state` + the placement layout (initial_map, retag, one
compute_conditions with the final states, prune_same_core_candidates) -
identical, cell by cell, to the old CPU-side preprocessing
(`test_init_map_from_state_matches_cpu_preprocessing`).
`Pyramid.onepass(mem)` = `init_map` + `upward` + `downward`. Former method names below (CoreItem.*, Pyramid instances,
`build`, `run`, `map_onepass`) refer to these.

**Renamed 2026-11-07:** `CoreItem.forced_by` is now `CoreItem.links`
(`update_forced_by` -> `update_links`), holding `(id_in, id_out, id_exit)`
triples - the exit link is the third item, so the separate `.exits` field
is gone; a dying in is `(id_in, DUMMY, DUMMY)`. The old `CoreItem.links`
(direct links from size-1 conditions) and `CoreItem.forces` were removed;
`candidate_pairs()` is now `candidate_links()`. Also removed the same day:
`.dependencies`, `.blocked`, `.conflicts` (see "Blocking - postponed"
below). Older items in this file still use the old names. `SquareItem.forced_by` (per-cell conditions) is
unchanged.

`CoreItem.forced_by`/`.placement_reach` id_out is now the LAST cell INSIDE
the tile that has a link pointing out (was: the first cell outside), and
every CoreItem keeps `.exits` = (last_inside, first_outside) links. A
parent joins child pairs directly (one child's id_out is the next child's
id_in), takes start points by the realm rule (id_in outside the parent, or
inside its own reporting child = a candidate), and records a pair when the
reached cell has an exit link out of the parent. Before this, every chain
was cut at the first border between two children. Tests:
`test_upstream_forced_by.py`; pictures: `upstream-forced-by.html` (this
convention) and `upstream-forced-by-current.html` (what the OLD code
produced on the same scenario). `update_aliases_down` now also informs the
sibling an exit link points into; `_consume_aliases_down` records the last
cell inside.

**Ins that die inside a tile (2026-11-07, Ivo: "a clear bug in the
upstream")**: a start whose chain forces a cell inside the tile but reaches
no exit is now kept as `(id_in, cores.DUMMY)` - at base level, in
`_compose_up` at every level above, and in `_consume_aliases_down`. Before,
it was dropped, and could never be connected further up (identified with a
candidate shared with another path, it can form new links through n-ary
conditions back at base level). On the swept maps 30-73% of parent pairs
are such dummies. A start with at least one exit gets no DUMMY, even if
another of its branches dies (C's (4,4) in the scenario) - decided by Ivo,
2026-11-07: "partly dying starts get no dummy".

Still open:
- **How to process DUMMY** (Ivo: "we then have to work out how to process
  dummy"). Nothing reads it yet beyond carrying it up.
- ~~`push_down` home child~~ - resolved 2026-11-07 (Ivo: "all candidates
  must be copied down, including those outside P, like Z"): `push_down` no
  longer copies the core's own triples to a child. It walks every
  `(candidate, cell)` the core received from its parent along the
  children's triples and tells every child the chain enters or passes
  through; each child derives its own triples for the candidate in the
  consume step. Test:
  `test_push_down_copies_an_outside_candidate_into_every_child_its_chain_touches`;
  picture: `downstream-z.html` (project root).
- **Blocking - postponed ("entirely for another day", Ivo, 2026-11-07)**:
  `CoreItem.dependencies`, `.blocked`, `.conflicts` and their update_*
  methods were removed for simplicity, with the 6 tests that only checked
  `update_blocked` (`bruteforce.unblock_path` itself is kept, now without a
  test). Points to remember for that day: a chain that enters a sibling
  and dies there forces cells no CoreItem list names (only its
  `(id_in, DUMMY, DUMMY)` survives); and between CoreItem children there
  is no way to tell whether they block each other.
- ~~Consumed facts lose candidate status going up~~ - resolved 2026-11-07:
  `.placement_reach` was removed. A candidate is whatever
  `cores.Pyramid.is_candidate` says (global placement-tile layout, every cell of a
  placement tile); `CoreItem.candidate_pairs()` is the candidate-started
  part of `.forced_by`, computed on demand, so pairs added on the way down
  count too.
- **Re-running the up pass from base erases the down pass** (base
  `update_forced_by` is a full recompute) - a second up pass must start one
  level above the children (see #3).

## 4. Possible explosion in `create_compound_ids` - reopened, the "capped
   at 3" assumption below no longer holds

Raised 2026-10-31: if many different placement tiles all contribute ids
that are mutually "syncable" (point 3 of the downstream plan - see #2
above), does enumerating every non-trivial subset (2^n - n - 1 for n ids)
blow up combinatorially, the way `compute_eligibility`'s own
flatten-and-substitute approach did (2026-10-14/15)?

Original preliminary analysis (2026-10-31, before implementation): likely
bounded, NOT exponential in the total number of placement tiles on the
board - reasoning rested on every vacuum condition being capped at 3
members (pre-reduction), giving exactly 4 subsets for 3 ids.

**Reopened 2026-11-01**: `create_compound_ids` (implemented, see #2 above)
is explicitly built to handle n > 3 too - Ivo, same day: "i still maintain
n-aries with n>3 matter" - confirmed working for n=4 (11 subsets) in
`test_create_compound_ids_enumerates_every_subset_size_two_and_up`.

**Why n > 3 genuinely matters, Ivo's own worked example (2026-11-01)**: on
one tile, 4 border ids id1/id2/id3/id4 (each from a different placement
tile, independently choosable - exactly create_compound_ids's own target
scenario). A border cell id5 has forced_by condition {id1, id2} (size 2).
A separate border cell id6 has forced_by condition {id3, id4} (size 2). An
interior cell id7 has forced_by condition {id5, id6} (size 2). Every
single condition here stays at size 2, never exceeding the real "<=3
members" cap - but id7 only ever actually becomes forced once id5 AND id6
are both chosen, and id5/id6 themselves each only become chosen once
BOTH of their own 2 members are chosen - so id7's real, transitive
dependency set is {id1, id2, id3, id4}, a genuine 4-ary fact ("a link is
created only if id1,id2,id3,id4 are chosen simultaneously" - Ivo's own
words), built entirely out of a CHAIN of size-2 conditions. The <=3 cap
bounds any single condition's own member count; it says nothing about
how large the compound group of ultimately-relevant border ids can grow
once several such conditions chain together - which is exactly why
create_compound_ids cannot be capped at n<=3 itself, even though the
individual board conditions it will eventually need to interact with
always are.

So the
"always <= 3 members" premise the original analysis leaned on no longer
holds as stated, so 2^n - n - 1 growth for a genuinely large n is back on
the table as an open risk, not a closed one - not yet a real problem
(n=4 is still trivially 11 subsets), but revisit the actual expected size
of n once step 2's "how to proceed" is decided and this gets a real
caller, rather than continuing to assume it stays small.

One thing that DOES need an explicit decision once this is implemented:
whether a virtual id can itself become a member of some FURTHER condition,
recursively (if so, that reintroduces something close to
compute_eligibility's own disaster - flattening-with-substitution across
multiple levels). Recommend deciding up front that virtual ids do NOT
recurse into further virtual-id creation, rather than discovering this the
hard way.

## 5. N-ary link generation during the passes - option 2 (decided 2026-11-08)

**Situation (Ivo, 2026-11-08):** base tiles now receive simultaneous
entries - several candidates whose chains enter the same tile (the down
pass delivers them: `(candidate, last cell, exit)` and
`(candidate, DUMMY, DUMMY)` triples in a base tile's `.links`). Entered
cells chosen together can complete an n-ary condition (a pair or triple
in some cell's `SquareItem.forced_by`) and so create a NEW single link,
from the compound of those candidates. The functions for this exist but
are untested in combination and not integrated:
`bruteforce.collapse_synced_ids`, `create_compound_ids`,
`generate_compound_candidates`, `prune_compound_candidates`,
`prune_blocked_compound_candidates`, `compoundize_forced_by`,
`propagate_compound_links` (see item 2). "Testing is a challenge, many
situations arise here and we have no overview or theoretical grasp."

**Three ways (Ivo):**
1. generate n-ary links once, on `.map_after` - OPTIMIZATIONS.md #7;
2. after EACH up step and EACH down step, take the new candidate links
   of that level down to base level, generate the n-ary links there, and
   climb back up to that level - (log n)^2 instead of log n, but easier
   to understand - **chosen, "for now we do 2"**;
3. move some n-ary information up, so links can be generated from one
   level's own data - OPTIMIZATIONS.md #8.

**Option 2, as a procedure (proposed, not yet built):** for the step at
level k (`Upstream.build_level(mem, k)` or `Downstream.down_level(mem, k)`):
1. Find the new candidate links of level k - the `candidate_links` that
   the step added.
2. Down: follow each to the base tiles its chain enters (the same
   walk-and-inbox mechanism as the down pass).
3. At base level: from each such tile's simultaneous entries, generate the
   n-ary links (compounds), and write them as new size-1 conditions into
   the cells' `SquareItem.forced_by` in `.map` (so everything downstream
   reads them the normal way).
4. Up: `Base.update_links` for the changed tiles, `Upstream.update_links`
   for their ancestors, up to level k.
5. Repeat 1-4 until the step adds no new candidate links (only additions
   ever happen, so this terminates).

**Open - to decide before building:**
- What a compound looks like in `.links`: as id_in of a triple (a
  `("compound", n)` id, as `get_or_create_compound` mints them)? What
  `is_candidate` says for it?
- How DUMMY is processed - a dying in is exactly a chain whose cells may
  complete an n-ary condition together with another candidate's.
- Whether a compound can itself become a member of a further compound
  (item 4 recommends NO, to avoid compute_eligibility's blow-up).

**Testing idea (proposed):** an independent oracle instead of
hand-enumerating situations - brute-force forward propagation on the whole
map: choose a set S of candidates (one per placement tile), mark them
chosen, apply every condition (single, pair, triple) to a fixed point, and
record the forced cells F(S). Then:
- soundness: every link the algorithm generates for a compound S forces
  only cells in F(S);
- completeness (small maps): for every small S (pairs, triples of
  candidates from different placement tiles), every cell in F(S) that no
  single member of S forces alone is explained by some generated link.
Run it on the hand-built scenarios first, then on swept maps.

**After n-ary links: blocking and unblocking (Ivo, 2026-11-08)** - the
next stage once item 5's link generation works. Blocking paths alone can
lead to situations where the placement options run out; then the
unblocking procedure has to step in. Blocking and unblocking are two more
sources of complexity - another reason for option 2: each up/down step is
followed by a full, easy-to-follow round trip to base level, into which
blocking and unblocking can later be hooked the same way as n-ary link
generation. Existing pieces to revisit then: `bruteforce.unblock_path`
(blocks a self-blocking cell, recomputes conditions), the old pipeline's
`closure.get_blocked_links` / `dissolve_blocked_paths`, the paused
kernel-style blocked-paths prototype in closure.py (memory note
c0-blocked-paths-mechanism-paused), and the CoreItem `.blocked` field
removed 2026-11-07 (item 3a, "Blocking - postponed").
