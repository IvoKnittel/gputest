"""Hierarchical pyramid over the base lattice, for the fixed-superlattice
closure refactor (see do_closure_old's own docstring for the problems
motivating it).

A patch id is a cell's own (row, col) index in the base lattice. A conflict
(id_a, id_b) records that two patches are mutually exclusive, stored low-id-
first so (a, b) and (b, a) are one entry in a set - unchanged, see
update_conflicts.

.dependencies replaces the earlier, too-simple .aliases: a set of
(cell_id, condition) pairs - condition one of cell_id's own
conditions.py-style triples (see bruteforce.compute_conditions) - rather than
a flat (id, id) pair. A plain pair lost exactly the information that matters:
which other ids would have to be chosen *together* to force cell_id, not just
that two ids are somehow related - see squareplacement_superlattice_refactor
memory note, "tile-ins need full .forced_by, not links."

A CoreItem is one node of the pyramid: a 3x3 array of elements - base-level
SquareItem for the lowest level, CoreItems above that - plus a margin of
read-only references around them, and the rectangular realm
((row_start, row_end), (col_start, col_end)) of base-lattice ids it owns.

Information flows both ways:
- up: .dependencies and .conflicts, what this core could not yet resolve
  inside its own realm (update_dependencies / update_conflicts).
- down: .aliases_down (populated 2026-10-25 by update_aliases_down - see its
  own docstring; the descending pass for .forced_by/.placement_reach
  specifically, informing a sibling child of cross-child reach that its own
  up-composed data alone can never reveal) and .blocked (still not built -
  .blocked is the set of path ids of this core that may not be placed in the
  current kernel call because they conflict with a path of another core
  placed simultaneously; it is recomputed per colorcode). .aliases_down is a
  one-shot inbox, not a permanent record: update_aliases_down immediately
  folds each informed child's own .aliases_down into that child's own
  .forced_by (via _consume_aliases_down) and clears it again, every call.

.forced_by (added 2026-10-22, redesigned 2026-10-23; .forced and .forces
also exist but are still unused): a set of plain (id_in, id_out) pairs,
id_out always external to this tile's own realm, id_in either external too
(a chain passes all the way through the tile) or, for a placement core, one
of its own PlacementStatus.candidate interior cells (a path originating
there, added 2026-10-23 - "for the placement tiles we also add to the list
paths that originate inside the tile") - see update_forced_by's own
docstring for the full reasoning and "how far chosen items on different
tiles can travel." Superseded its own first version (2026-10-22, a
candidate-filtered set of (cell_id, condition) n-ary pairs, same shape as
.dependencies) same day, after Ivo clarified the real goal was pure
single-link cross-tile reach, not n-ary conditions. .forces' own redesigned
meaning under this scheme is not yet decided (confirmed 2026-10-23: "of no
use" for now) - don't assume it mirrors .forced_by's old reverse-index
relationship from the superseded version.

.links (added 2026-10-23): old-sense direct (forced_id, forcer_id) links -
.forced_by1's own binary relation, re-derived from a size-1 (post-reduction)
.forced_by condition (see update_links). Computed per tile on its own, then
assembled into the whole network's links by one pass up the hierarchy (plain
union at every level, no realm filtering) - a link is a settled local fact,
unlike .dependencies/.conflicts, which still carry a realm-relative
qualifier. update_forced_by's own per-tile edges are built the same way,
just not stored as .links itself.

.placement_reach (added 2026-10-25): a set of (candidate_id, id_out) pairs -
the subset of what .forced_by describes whose origin is confirmed traceable
to a genuine PlacementStatus.candidate cell, kept as its own field rather
than folded into .forced_by (which also carries origins that are just
arbitrary free cells, not yet traceable to any placement decision - see
update_placement_reach's own docstring and the squareplacement_superlattice_
refactor memory note, 2026-10-25 entry, on how much of a typical .forced_by
turns out to be exactly that: dangling, not yet "live").

.forced_by and .placement_reach both now follow the hierarchy UP (one level
at a time, via _compose_up - elements-are-CoreItem branch added 2026-10-25,
"did you already follow the hierarchy up and down" - up: yes, as of now;
down: still not built for anything, same as .aliases_down/.blocked above).
"""
from map_of_squares import PlacementStatus


class CoreItem:
    """elements: 3x3 object array (base-level SquareItem or CoreItem, never
    mixed). margin: the read-only items around them. realm: the rectangle of
    base-lattice ids this core owns.
    """
    def __init__(self, elements, margin, realm):
        self.elements = elements
        self.margin = margin
        self.realm = realm
        self.dependencies = set()
        self.conflicts = set()
        self.aliases_down = set()
        self.blocked = set()
        self.forced = set()
        self.forced_by = set()
        self.forces = set()
        self.links = set()
        self.placement_reach = set()

    def update_dependencies(self):
        """Recompute .dependencies from the elements.

        At the base level (elements are SquareItem), for every element's own
        absolute (row, col) id and every one of its .forced_by, keep the
        pair (cell_id, condition) whenever the condition has at least one
        member outside this core's own realm - i.e. exactly
        vacuum.boundary_crossing's own rule, run here instead of recomputed
        from scratch at every level.

        One level up (elements are CoreItem), union every child's own
        .dependencies (already (cell_id, condition) pairs, relative to each
        child's own smaller realm) and keep only the pairs whose condition
        still reaches outside *this* bigger realm - confirmed (see
        squareplacement_superlattice_refactor memory note) to give exactly
        the same result as recomputing from the base lattice directly at this
        realm.
        """
        first = self.elements.flat[0]
        if isinstance(first, CoreItem):
            merged = set()
            for child in self.elements.flat:
                merged |= child.dependencies
            self.dependencies = {(cell_id, condition) for cell_id, condition in merged
                                  if not all(in_realm(member_id, self.realm)
                                             for member_id, _status in condition)}
        else:
            (row_start, _), (col_start, _) = self.realm
            rows, cols = self.elements.shape
            collected = set()
            for li in range(rows):
                for lj in range(cols):
                    cell_id = (row_start + li, col_start + lj)
                    item = self.elements[li, lj]
                    for condition in item.forced_by:
                        if not all(in_realm(member_id, self.realm)
                                   for member_id, _status in condition):
                            collected.add((cell_id, condition))
            self.dependencies = collected

    def update_conflicts(self):
        """Recompute .conflicts from the elements: every element's conflicts,
        minus the pairs whose two ids both lie inside this core's realm -
        unchanged since .conflicts is still a flat (id, id) pair, left as-is.
        """
        merged = set()
        for element in self.elements.flat:
            if isinstance(element, CoreItem):
                merged |= element.conflicts
            else:
                raise NotImplementedError(
                    "update_conflicts for base-level elements "
                    f"({type(element).__name__}) is not defined yet")
        self.conflicts = {pair for pair in merged
                           if not (in_realm(pair[0], self.realm) and in_realm(pair[1], self.realm))}

    def _elements_by_id(self):
        """Base-level-only helper: {absolute cell_id: SquareItem} for every
        one of this core's own elements - raises NotImplementedError if
        elements are themselves CoreItems (same precedent as
        update_conflicts leaving that level unimplemented until needed).
        Shared by update_forced_by and update_forces, both of which need to
        look an id up by its absolute (row, col) position.
        """
        first = self.elements.flat[0]
        if isinstance(first, CoreItem):
            raise NotImplementedError(
                "this operation is not defined yet for CoreItem-level elements")
        (row_start, _), (col_start, _) = self.realm
        rows, cols = self.elements.shape
        return {(row_start + li, col_start + lj): self.elements[li, lj]
                for li in range(rows) for lj in range(cols)}

    def _base_edges(self):
        """Base-level-only helper, shared by update_forced_by and
        update_placement_reach: builds the two kinds of single-link edges
        from this tile's own elements, plus a placement_status lookup for
        every id touched.

        - forward_edges: forcer_id -> set of cell_id, from each interior
          cell's own size-1 .forced_by condition {forcer_id} (forcer_id may
          be interior or external).
        - exit_edges: interior_cell -> set of external b, from each interior
          cell's own .forces, wherever (b, conditions) has b outside this
          tile and at least one condition in conditions is size 1 (this
          interior cell is b's sole remaining forcer for that condition).
        - status_of: id -> PlacementStatus, for every id that appeared as a
          forcer_id above (tag carried on the (id, status) condition member
          itself - see SquareItem.forced_by's own docstring) or as one of
          this tile's own cells (read directly off its .placement_status).

        No margin or whole-map access needed - SquareItem.forced_by/.forces
        are already global fields (built over the whole board by
        conditions.initial_map/bruteforce.update_forces, not scoped to any
        tile), so an interior cell's own .forces already names any external
        cell it directly forces, and a condition's own tags already carry
        every member's placement_status, with no need to look outside
        self.elements.
        """
        by_id = self._elements_by_id()
        interior_ids = set(by_id)

        forward_edges = {}
        status_of = {}
        for cell_id, item in by_id.items():
            status_of[cell_id] = item.placement_status
            for condition in item.forced_by:
                if len(condition) == 1:
                    forcer_id, forcer_status = next(iter(condition))
                    forward_edges.setdefault(forcer_id, set()).add(cell_id)
                    status_of.setdefault(forcer_id, forcer_status)

        exit_edges = {}
        for cell_id, item in by_id.items():
            for b, conditions in item.forces:
                if b in interior_ids:
                    continue
                if any(len(c) == 1 for c in conditions):
                    exit_edges.setdefault(cell_id, set()).add(b)

        return interior_ids, forward_edges, exit_edges, status_of

    @staticmethod
    def _reach(origin, start_nodes, interior_ids, forward_edges, exit_edges):
        """Breadth-first walk from start_nodes through interior_ids-only
        waypoints (via forward_edges), recording (origin, out_id) whenever a
        visited node has its own exit_edges entry. Shared base-level BFS
        core for update_forced_by and update_placement_reach - see
        _base_edges's own docstring for what the three edge/lookup
        structures mean.
        """
        frontier = {node for node in start_nodes if node in interior_ids}
        visited = set()
        found = set()
        while frontier:
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for out_id in exit_edges.get(node, ()):
                    found.add((origin, out_id))
                for nxt in forward_edges.get(node, ()):
                    if nxt in interior_ids and nxt not in visited:
                        next_frontier.add(nxt)
            frontier = next_frontier
        return found

    def update_forced_by(self):
        """Recompute .forced_by. Redesigned 2026-10-23 (supersedes the
        candidate-filtered (cell_id, condition) version this held
        2026-10-22..23): now a set of plain (id_in, id_out) pairs - "we need
        only one item to force another item." id_out is always external to
        this core's own realm. id_in is either external too (a chain passes
        all the way through the tile, entering and leaving - "how far the
        chosen items on different tiles can travel") or, for a placement
        core specifically, one of its own PlacementStatus.candidate interior
        cells (a path originating there, added 2026-10-23 - "for the
        placement tiles we also add to the list paths that originate inside
        the tile"): since a placement core's own interior cells are genuine
        future placement decisions (unlike a transit core's, which are only
        ever relays), a path starting there and reaching an external id is
        as real a fact as one entering from outside. A transit core has no
        candidate cells of its own, so this adds nothing there - no
        branching on tile kind is needed, the candidate check alone decides
        it.

        At the base level (elements are SquareItem; see _base_edges), a
        breadth-first walk (_reach) from (a) every external id with a direct
        entry edge into the tile, and (b) every one of this tile's own
        candidate cells, collects every external id_out reachable via some
        interior cell's own exit edge. A candidate-origin pair can never be
        a self-loop (id_in == id_out) by construction - exit_edges only ever
        names external ids, and a candidate origin's own id is, by
        definition, interior.

        One level up (elements are CoreItem; added 2026-10-25, corrected same
        day - "did you already follow the hierarchy up" - this hadn't, until
        now): .forced_by is the union of two differently-composed parts, not
        one plain _compose_up call on the raw union of children's pairs -
        that first attempt was wrong, caught by cross-checking a composed
        multi-level result against a single CoreItem built directly over the
        same bigger realm (see squareplacement_superlattice_refactor memory
        note, 2026-10-25 entry, for the worked example): a child's own
        non-candidate external origin does NOT remain a valid origin once it
        becomes interior to a bigger realm (it was only ever "external" to
        that one small child, not a real placement decision) - only a
        genuine candidate origin survives regardless of realm size, which is
        exactly what .placement_reach already tracks correctly. So: (a) the
        union of children's own .forced_by, composed via _compose_up with
        require_external_origin=True - re-filters origins fresh at this
        level, keeping only ids still external to THIS bigger realm, and
        (b) self.placement_reach (update_placement_reach must be called
        first - same children-before-parent discipline every recursive
        field here already needs), which already correctly carries every
        candidate-origin pair regardless of realm size. Their union is
        .forced_by.
        """
        first = self.elements.flat[0]
        if isinstance(first, CoreItem):
            child_pairs = set()
            for child in self.elements.flat:
                child_pairs |= child.forced_by
            external_part = _compose_up(child_pairs, self.realm, require_external_origin=True)
            self.forced_by = external_part | self.placement_reach
            return

        interior_ids, forward_edges, exit_edges, status_of = self._base_edges()

        results = set()

        external_ins = {fid for fid in forward_edges if fid not in interior_ids}
        for id_in in external_ins:
            results |= self._reach(id_in, forward_edges[id_in], interior_ids, forward_edges, exit_edges)

        candidate_starts = {cid for cid in interior_ids if status_of[cid] == PlacementStatus.candidate}
        for cell_id in candidate_starts:
            results |= self._reach(cell_id, {cell_id}, interior_ids, forward_edges, exit_edges)

        self.forced_by = results

    def update_placement_reach(self):
        """Recompute .placement_reach (added 2026-10-25): a set of plain
        (candidate_id, id_out) pairs, candidate_id always a genuine
        PlacementStatus.candidate cell SOMEWHERE on the board (not
        necessarily inside this tile), id_out always external to this
        core's own realm. "The most important thing to know now are for
        each tile the entries that originate from placement tiles" - the
        subset of .forced_by's own (id_in, id_out) pairs whose id_in is
        itself traceable to a placement core, kept as its own separate field
        rather than folded into .forced_by's own broader (and mostly
        dangling - see squareplacement_superlattice_refactor memory note,
        2026-10-25 entry) id_in universe.

        At the base level (elements are SquareItem; see _base_edges), the
        candidate origins available to THIS tile are (a) its own interior
        cells tagged candidate, and (b) any EXTERNAL id that directly enters
        this tile (appears in forward_edges) and is ITSELF tagged candidate
        in the condition that named it (status_of, from _base_edges) - no
        extra lookup needed, since a condition's own (id, status) tag
        already carries this for every member it has ever touched, not just
        this tile's own cells. From each such origin, the same _reach walk
        as update_forced_by's own collects every external id_out reachable.

        One level up (elements are CoreItem), .placement_reach is the union
        of every child's own .placement_reach, composed through this bigger
        realm via _compose_up - identical mechanism to update_forced_by's
        own recursive case, since every child's own pair already has a
        confirmed-candidate origin and composition preserves that origin
        unchanged; no extra candidate-checking is needed above the base
        level.
        """
        first = self.elements.flat[0]
        if isinstance(first, CoreItem):
            child_pairs = set()
            for child in self.elements.flat:
                child_pairs |= child.placement_reach
            self.placement_reach = _compose_up(child_pairs, self.realm)
            return

        interior_ids, forward_edges, exit_edges, status_of = self._base_edges()

        results = set()
        candidate_ids = {cid for cid, status in status_of.items()
                          if status == PlacementStatus.candidate}
        for cid in candidate_ids:
            start_nodes = {cid} if cid in interior_ids else forward_edges.get(cid, set())
            results |= self._reach(cid, start_nodes, interior_ids, forward_edges, exit_edges)

        self.placement_reach = results

    def update_aliases_down(self):
        """The descending pass (added 2026-10-25 - "we need to follow
        through links between child supertiles of the same level"): informs
        each sibling child about reach that originates in a DIFFERENT
        sibling's own placement-core territory, which update_placement_reach's
        own up-composition throws away - a chain landing inside ANY sibling
        (not genuinely external to the whole parent) is just an anonymous
        relay waypoint there, never reported as a fact about that specific
        sibling. Example (Ivo's own): on a supertile with a 3x3 grid of
        children, (1,0) is a placement tile (has its own candidate cell(s));
        is there a path from (1,0) reaching all the way into (1,2)'s own
        territory (passing through (1,1) along the way)? That fact is real
        and discoverable from the union of the children's own data, but
        neither (1,0) nor (1,2) could find it alone - (1,0) only knows it
        exits its OWN realm somewhere, not which sibling that lands in;
        (1,2) has no way to know (1,0) exists at all.

        elements-are-SquareItem (base) level: does nothing (no siblings to
        inform - a base tile has no CoreItem children).

        elements-are-CoreItem level, two different child fields used for two
        different purposes (corrected 2026-10-25, same day as the first
        version - see Ivo's own follow-up worked example):
        - ORIGINS (where a walk may start) are confirmed candidates only -
          the union of every child's own .placement_reach keys - same
          reasoning as update_placement_reach's own base case: "is there a
          path FROM (1,0)" means from (1,0)'s own placement decisions
          specifically, not from an arbitrary cell that merely happens to
          have an edge.
        - EDGES (how a walk may continue once started) are built from the
          union of every child's own .forced_by, not just .placement_reach.
          .placement_reach is always a subset of .forced_by (verified
          elsewhere), so this is a strict widening, not a different set:
          once a walk reaches some cell b that a DIFFERENT sibling already
          recognizes as one of its own valid .forced_by entries (whether b
          got there as a plain external entry or as that sibling's own
          candidate), the walk can continue through b's own further reach,
          even though b itself was never a candidate. Concretely: if a
          foreign candidate a's walk touches sibling (0,0) at cell b
          (recorded, informing (0,0)), and (0,0)'s own .forced_by already
          contains (b, c) with c inside sibling (1,0), then a's reach
          continues on to c too, and (1,0) gets informed - (0,0) is not a
          dead end just because b itself isn't a's own interior cell or a
          candidate. The first version of this method only used
          .placement_reach for edges too, which missed exactly this case -
          corrected the same day it was found, via Ivo's own follow-up
          example.

        For every confirmed-candidate origin, walk forward through this
        edges graph exactly as _compose_up does, but instead of only
        recording the final exit from the WHOLE parent realm, record
        (origin, node) into the OWNING sibling's own .aliases_down every
        time the walk visits a node that falls inside a sibling OTHER than
        the origin's own - informing that sibling of reach it could never
        have discovered locally. Every child's own .aliases_down is cleared
        and fully recomputed each call (same "full recompute, not
        incremental" discipline as every other update_* method here).

        Does not itself recurse into climbing further down (clearing
        grandchildren's own .aliases_down, say) - this call only ever
        touches its own direct children, one level down, same as every
        other update_* method here only ever touches its own direct
        children or elements, never further.

        At the end of informing every child, each informed child's own
        .aliases_down is immediately CONSUMED into its own .forced_by, then
        cleared (2026-10-25 - "At the end, you should update the CoreItems
        .forced_by and clear the .aliases_down list"): see
        _consume_aliases_down's own docstring for how. .aliases_down is a
        one-shot inbox, not a permanent record - once its facts are folded
        into .forced_by, nothing further needs it until the next round's
        own update_aliases_down call repopulates it.
        """
        first = self.elements.flat[0]
        if not isinstance(first, CoreItem):
            return

        children = list(self.elements.flat)
        for child in children:
            child.aliases_down = set()

        origin_pairs = set()
        edge_pairs = set()
        for child in children:
            origin_pairs |= child.placement_reach
            edge_pairs |= child.forced_by

        edges = {}
        for a, b in edge_pairs:
            edges.setdefault(a, set()).add(b)

        def owning_child(id_):
            for child in children:
                if in_realm(id_, child.realm):
                    return child
            return None

        origin_owner = {a: owning_child(a) for a, _b in origin_pairs}

        for origin in {a for a, _b in origin_pairs}:
            src_child = origin_owner[origin]
            frontier = {origin}
            visited = set()
            while frontier:
                next_frontier = set()
                for node in frontier:
                    if node in visited:
                        continue
                    visited.add(node)
                    owner = owning_child(node)
                    if owner is not None and owner is not src_child:
                        owner.aliases_down.add((origin, node))
                    for nxt in edges.get(node, ()):
                        if in_realm(nxt, self.realm) and nxt not in visited:
                            next_frontier.add(nxt)
                frontier = next_frontier

        for child in children:
            child._consume_aliases_down()

    def _consume_aliases_down(self):
        """Fold this core's own .aliases_down into its own .forced_by, then
        clear .aliases_down - called by the PARENT's own update_aliases_down,
        once per informed child, right after populating .aliases_down (see
        its own docstring, "At the end...").

        For every (origin, b) in .aliases_down - b one of this core's own
        cells, reached by origin via a chain through some OTHER sibling -
        treats b exactly like any other newly-discovered entry point: walks
        forward from b through this core's own interior (reusing the same
        edge-building its own update_forced_by already uses - _base_edges/
        _reach at the base level, or the union of its own children's
        .forced_by at a recursive level, same graph _compose_up itself
        would build), and adds (origin, exit) to .forced_by for every exit
        this core's own interior lets b's chain reach - origin, not b,
        since b is only ever a relay waypoint, the same discipline
        update_forced_by's own candidate-origin case already follows.

        No-ops the .forced_by extension (but still clears .aliases_down) if
        this core's own .elements is None - a real CoreItem always has real
        elements; None only ever occurs in this file's own synthetic unit
        tests of update_aliases_down's informing logic in isolation, which
        have nothing to extend through.
        """
        if not self.aliases_down:
            return

        if self.elements is None:
            self.aliases_down = set()
            return

        first = self.elements.flat[0]
        if isinstance(first, CoreItem):
            edges = {}
            for child in self.elements.flat:
                for a, b in child.forced_by:
                    edges.setdefault(a, set()).add(b)
            for origin, b in self.aliases_down:
                frontier = {b}
                visited = set()
                while frontier:
                    next_frontier = set()
                    for node in frontier:
                        if node in visited:
                            continue
                        visited.add(node)
                        for nxt in edges.get(node, ()):
                            if in_realm(nxt, self.realm):
                                if nxt not in visited:
                                    next_frontier.add(nxt)
                            else:
                                self.forced_by.add((origin, nxt))
                    frontier = next_frontier
        else:
            interior_ids, forward_edges, exit_edges, _status_of = self._base_edges()
            for origin, b in self.aliases_down:
                self.forced_by |= self._reach(origin, {b}, interior_ids, forward_edges, exit_edges)

        self.aliases_down = set()

    def update_links(self):
        """Recompute .links: old-sense direct links, (forced_id, forcer_id)
        pairs where forcer_id alone, once chosen, forces forced_id - exactly
        the old pipeline's own binary .forced_by1 relation (see
        map_of_squares.SquareItem's own docstring), re-derived here from the
        n-ary mechanism's post-reduction single-element conditions (see
        bruteforce.compute_conditions's own triple->pair->single reduction).

        Unlike update_dependencies/update_forced_by/update_forces, this is
        computed per tile on its own and then assembled into the whole
        network by one pass through the hierarchy, not filtered by realm at
        each level - a link, once formed, is already a settled, local fact
        about two specific cells; climbing the pyramid only ever adds
        non-overlapping territory, it never invalidates a child's own link.

        At the base level (elements are SquareItem): every one of this
        core's own cells' current .forced_by conditions of size 1
        contributes (forced_id, forcer_id) - forcer_id the condition's own
        lone member id, its tag discarded (a link only needs *which* id,
        the old .forced_by1 relation carried no placement_status either).

        One level up (elements are CoreItem): .links is just the union of
        every child's own .links - "assemble the whole network by one pass
        through the hierarchy."
        """
        first = self.elements.flat[0]
        if isinstance(first, CoreItem):
            merged = set()
            for child in self.elements.flat:
                merged |= child.links
            self.links = merged
        else:
            by_id = self._elements_by_id()
            self.links = {
                (cell_id, next(iter(condition))[0])
                for cell_id, item in by_id.items()
                for condition in item.forced_by
                if len(condition) == 1
            }


def in_realm(id_, realm):
    """True if id_ (a (row, col) base-lattice index) falls inside realm -
    ((row_start, row_end), (col_start, col_end)), both ends exclusive on
    row_end/col_end, matching image_to_squares.core_range_for_tile.
    """
    (row_start, row_end), (col_start, col_end) = realm
    row, col = id_
    return row_start <= row < row_end and col_start <= col < col_end


def _compose_up(child_pairs, realm, require_external_origin=False):
    """Compose a set of children's own (id_in, id_out) pairs (same shape as
    CoreItem.forced_by/.placement_reach) into the bigger realm's own
    version of the same field - shared by both fields' recursive
    (elements-are-CoreItem) branch (added 2026-10-25, "follow the hierarchy
    up").

    child_pairs, unioned across all of this core's own children, already
    describes a directed graph: id_in reaches id_out in one step. An id that
    was external to one specific child can now be internal to this bigger
    realm (it belongs to a sibling child) - in that case the chain
    continues through it (using whatever further pairs THAT id has, from
    whichever child(ren) established them) rather than stopping there,
    since a sibling's own pairs already capture every entry-to-exit path
    through its own interior exhaustively (recomputing that internal detail
    here would be redundant, not more correct).

    require_external_origin controls which ids from child_pairs are walked
    again as origins here - a correction (2026-10-25, same day) to this
    function's own first version, which treated every id that was EVER an
    origin for ANY child as unconditionally still an origin here. That is
    right for a confirmed-candidate origin (candidate status never changes
    with realm size - see update_placement_reach, which is NOT affected by
    this parameter and always wants the unconditional behaviour, the
    default False), but wrong for a plain external-entry origin: a
    non-candidate id that was merely external to one small child can become
    interior to this bigger realm (it belongs to a sibling child now folded
    in) - at that point it is not a placement decision and not external to
    anything anymore, so it should stop being tracked as a source, not keep
    being one just because some smaller child once saw it that way. Caught
    by cross-checking a composed multi-level .forced_by against a single
    CoreItem built directly over the same bigger realm - the unconditional
    version produced extra, spurious pairs the direct computation did not
    (see squareplacement_superlattice_refactor memory note, 2026-10-25
    entry, for the worked example). With require_external_origin=True,
    origins are re-filtered to exactly the ids NOT in_realm(this realm);
    CoreItem.update_forced_by then unions in self.placement_reach itself to
    recover the candidate-origin half, rather than this function trying to
    rediscover candidate status (which it has no way to check) on its own.

    Every surviving origin is walked breadth-first through in_realm(this
    realm) waypoints, recording (origin, out_id) for every out_id reached
    that is NOT in_realm(this realm).
    """
    edges = {}
    for a, b in child_pairs:
        edges.setdefault(a, set()).add(b)

    if require_external_origin:
        origins = {a for a, _b in child_pairs if not in_realm(a, realm)}
    else:
        origins = {a for a, _b in child_pairs}
    results = set()
    for origin in origins:
        frontier = {origin}
        visited = set()
        while frontier:
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for nxt in edges.get(node, ()):
                    if in_realm(nxt, realm):
                        if nxt not in visited:
                            next_frontier.add(nxt)
                    else:
                        results.add((origin, nxt))
            frontier = next_frontier
    return results
