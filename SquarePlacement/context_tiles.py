"""The tile/supertile closure algorithm, done with context - replacing the
vacuum-flattening approach in bruteforce.compute_eligibility, which blew up
combinatorially (see squareplacement_superlattice_refactor memory note) for
a specific, now-understood reason: it assumed "every group of cells outside
may be selected together" and tried to fully resolve that assumption in one
eager pass, starting from nothing. That situation - maximal uncertainty, no
prior structure - never actually arises in real operation, and is precisely
where the mechanism is pathological.

What's actually known at every real step of the procedure (recalled, not
re-derived, 2026-10-15):

1. Before a round's placements, the whole map's current state is known
   completely - not just locally per core, but its full global connectivity
   (the .forced_by/.forces graph as it currently stands) too.
2. Where the next round's simultaneous placements will happen is known in
   advance - one placement per active core of the current supergrid sweep,
   active cores separated by three non-placement cores (see
   image_to_squares.py's own tile_upper_left_indices/colorcode scheme).
3. A core's "ins" and "outs" (see squareplacement_superlattice_refactor
   memory note, "patch2's cross-tile boundary rule") computed from that
   current state are stale by the time placements happen - but still real
   structure: the topology the placements act on, not built from nothing.
4. A higher-order term (e.g. an (a, b, c) alias/conflict relevant only once
   aggregated far enough up the hierarchy) is only ever relevant if a chain
   of ordinary (a, x0, x1)-type links already connects things. A placement
   can close a gap in that existing chain and make it suddenly relevant -
   but the chain, and the gap, have to already exist. It never happens
   starting from a vacuum.

The base per-cell .forced_by/.forces structure (conditions.py/bruteforce.py)
and the realm/pyramid machinery (cores.py) are still the right building
blocks - nothing about *those* was wrong. What has to change is not reusing
them to eagerly flatten everything from a standing start; instead, update
tiles and supertiles using the existing (stale but real) structure and the
specific placements that just happened, not a context-free re-derivation.

Starting empty on purpose: the right shape for this isn't settled yet - see
test_context_tiles.py, which drives this file by exploring concrete small
simultaneous-placement situations first, to find out what's actually safe to
assume in general before committing to an algorithm.
"""
