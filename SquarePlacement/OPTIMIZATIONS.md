# Optimization candidates

Tracking list for performance/efficiency work identified while building the
condition-based closure redesign (conditions.py/bruteforce.py/cores.py).
None of these are implemented - explicit, deliberate "don't care about
efficiency yet" stance throughout the exploratory phase (see
squareplacement_superlattice_refactor memory note). Listed here so they
don't need re-discovering later.

## Candidates

1. **`cores.CoreItem.update_blocked`'s candidate list isn't state-filtered**
   (found 2026-10-29). `candidates` is built purely from `.placement_status`
   (a fixed, geometric fact - "is this cell inside a placement core"), not
   also filtered by `.state == StateEnum.free`. Once a cell becomes
   non-free (e.g. via `bruteforce.unblock_path`), it can still appear as
   `id2` in other candidates' own `.blocked` pairs (confirmed harmless: the
   underlying exclusion fact stays true, just moot once the cell is already
   excluded for a different, settled reason - not a correctness bug, just
   wasted work/clutter). Candidate fix: skip non-free cells both as `id1`
   and `id2` in `update_blocked`'s own loop.

2. **`bruteforce.compute_conditions`'s full-board, non-incremental rescan**
   (profiled 2026-10-25) - the dominant cost by far once boards get
   realistically large: ~1.2-1.8s per call on a 63x63 board (441 tiles, 25
   placement cores), rescanning all ~3969 cells even when a single round
   only changed ~25 of them (0.6% of the board). Natural fix: incrementality
   - only reprocess cells whose conditions could plausibly have changed this
   round (e.g. within some radius of this round's actual placements),
   rather than the whole board every call.

3. **The CoreItem pyramid rebuild has the same full-recompute-every-round
   pattern** as #2 (`update_forced_by`/`update_placement_reach`/
   `update_blocked`/`update_aliases_down`, profiled 2026-10-25) - smaller
   absolute cost at current scale (~10-15x cheaper than compute_conditions,
   since it only re-reads already-reduced per-tile condition sets rather
   than re-deriving them from scratch), but would scale the same way
   (worse) on a bigger board. No composition-algorithm fix needed (the
   `_compose_up`/`_reach` BFS itself is already cheap and non-redundant,
   confirmed by profiling and by instrumenting for duplicate calls) - the
   cost is purely from rebuilding every tile from scratch every round
   rather than only the ones that actually changed.

4. **`bruteforce.init_vacuum`'s own per-cell materialization cost**
   (profiled 2026-10-14) - the remaining dominant cost after caching
   `vacuum_condition_offsets`/`_forces_offsets` (~5.85s self-time for 6561
   cells on an 81x81 board). Materializing ~76 actual translated
   (absolute-coordinate) frozensets per cell is genuinely distinct work for
   each cell - caching only removed the *shared* grouping work around it,
   not the per-cell materialization itself. Further speedup would need a
   different representation (storing/working with offsets instead of
   materializing absolute-coordinate frozensets per cell) - a bigger
   architectural change, not attempted.

5. **`bruteforce.retag_forced_by`/`update_forces` are also full, not
   incremental** (same class as #2/#3, noted 2026-10-14/22) - every call
   re-tags/rebuilds every cell's own data from scratch over the whole map,
   relying on callers to invoke it explicitly whenever `.forced_by`/
   `.placement_status` change, rather than tracking what actually changed
   and updating only that.

6. **`.forces` kept on a supersupertile-scoped window rather than a precise
   margin-3 one** (decided 2026-10-31, interim - see TODO.md item 0). Unlike
   `.forced_by` (`conditions.forced_by_on_realm`, reach 2, pure geometry +
   the true board shape, no real neighbour data needed), correctly
   maintaining `.forces` as `.forced_by` gets mutated by real placement
   decisions needs `bruteforce.update_forces`'s own whole-array inversion
   over REAL, CURRENT neighbour data within reach 3 - genuinely requires a
   loaded window, not just the true shape. Rather than building a precise
   margin-3 window loader now, the interim choice is to run
   `update_forces` scoped to a level-2 CoreItem's own realm (a "supersuper-
   tile" - 3x3 of 3x3 of base tiles, same terminology as Quality/
   test_image_to_squares.py's superlattice/supersuperlattice sweeps) around
   whatever core needs its `.forces` refreshed: comfortably bigger than
   reach 3 for any cell away from the supersupertile's own outer edge, at
   the cost of redoing far more work than the minimal margin actually
   needs. Revisit with a precise margin-3 `forces_on_realm`-style window
   once this becomes worth the precision - the 9x (or worse) oversized
   recompute is the accepted cost for now.

## Shared theme

#2, #3, #5 are all the same underlying pattern: a deliberate "full
recompute, not incremental" discipline, chosen throughout this whole
redesign for simplicity/correctness while the algorithm itself was still
being worked out (see memory note, "Ivo knows the brute-force/vacuum
exploration ... scales horribly and has no future ... staying with it
anyway ... to get aligned on terminology and algorithmic understanding
... before moving to whatever the actually-scalable version looks like").
None of these are bugs - they're accepted tradeoffs for the exploratory
phase, revisit only if/when this becomes a real bottleneck for a
production-scale board or a long run of many rounds.
