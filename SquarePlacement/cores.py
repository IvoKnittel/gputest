"""Hierarchical pyramid over the base lattice, for the fixed-superlattice
closure refactor (see do_closure_old's own docstring for the problems
motivating it).

A patch id is a cell's own (row, col) index in the base lattice.

Structure (2026-11-07, Ivo: "Create a class Pyramid, with classes Upstream,
Downstream as members ... Create a class Base and make all functions
processing SquareItems its members. Base also contains the map as a
member."):

- CoreItem: one node of the pyramid, data only - a 3x3 array of elements
  (SquareItem for a base tile, CoreItems above that), a margin of read-only
  references around them, the rectangular realm ((row_start, row_end),
  (col_start, col_end)) of base-lattice ids it owns, and .links.
- Base: holds the map (.map) and everything that works on SquareItems -
  generating the base-level CoreItems (generate_core_items), reading their
  single links, and the base-level halves of the up and down steps. After
  a down pass it also holds .map_onepass, the updated base level.
- Upstream: the up step for a CoreItem whose children are CoreItems, and
  build(base_items), the whole up pass; holds the hierarchy as .pyramid.
- Downstream: the down step for a CoreItem whose children are CoreItems
  (update_aliases_down, the CoreItem half of the consume step, push_down),
  and run(pyramid), the whole down pass.
- Pyramid: owns one of each (.base, .upstream, .downstream), runs them in
  order (onepass), and holds the functions both directions share.

Flow (Ivo, 2026-11-07: "Base initializes the map and generates CoreItems
from SquareItems. The result is fed into Upstream. Upstream also holds the
CoreItem hierarchy as a list .pyramid. Reaching the top, .pyramid is given
to Downstream. After updating the lowest CoreItem level Downstream writes
the lowest-level array back to Base into .map_onepass"):

    base_items = base.generate_core_items()   # level 0, .links computed
    upstream.build(base_items)                # upstream.pyramid, base .. top
    downstream.run(upstream.pyramid)          # top .. base; sets base.map_onepass

Functions that used to branch on "elements are SquareItem / CoreItem" are
now two functions, one in Base and one in Upstream/Downstream; the caller
picks by level (level 0 = base tiles).

.links (named .forced_by until 2026-11-07; .forced also exists but is
still unused): a set of (id_in, id_out, id_exit) triples - a chain of
single links enters the tile from id_in, or starts at candidate id_in
inside it ("for the placement tiles we also add to the list paths that
originate inside the tile", 2026-10-23); id_out is the LAST cell INSIDE
the tile with a link pointing out ("last cell inside", 2026-11-07), and
id_exit the cell outside the tile that link points to - one triple per
exit link, so an id_out linking out in two places gives two triples. A
chain that forces a cell inside but reaches no exit is (id_in, DUMMY,
DUMMY). (Until 2026-11-07 the exit links were a separate .exits field; the
third item replaces it.)

History of the field: first (2026-10-22) a candidate-filtered set of
(cell_id, condition) n-ary pairs, superseded the same day after Ivo
clarified the real goal was pure single-link cross-tile reach, not n-ary
conditions; then (id_in, first cell outside) pairs; then (id_in, last cell
inside) pairs plus .exits. Removed on 2026-11-07: CoreItem.forces (never
filled), .placement_reach (always the candidate-started part - see
Pyramid.candidate_links), .exits (now the third item), the old .links field
(direct (forced_id, forcer_id) links from size-1 conditions; nothing read
it), and .dependencies, .blocked and .conflicts with their update_* methods
(Ivo: "for simplicity. This is entirely for another day"). SquareItem's own
.forced_by / .forces, the per-cell conditions, are unaffected.

A candidate is whatever Pyramid.is_candidate says (the global
placement-tile layout - every cell of a placement tile).
"""
import numpy as np


# id_out and id_exit of a .links triple whose chain dies inside the tile
# (added 2026-11-07, Ivo: "ins that die within P need to be in P's
# .forced_by ... Put them on the list as (id, dummy)"; .forced_by is now
# .links, and the dummy fills both places). (id_in, DUMMY, DUMMY) means: a
# chain enters from id_in (or starts at candidate id_in), forces at least
# one cell inside the tile, and reaches no exit. Such an in can still
# matter further up - identified there with a candidate shared with another
# path, it can form new links through n-ary conditions back at base level.
# How DUMMY is processed further is still open. DUMMY is not a cell: never
# pass it to in_realm / is_candidate / owning_child.
DUMMY = "dummy"


# Which base tiles are placement tiles (added 2026-11-07). Base tile (ci, cj)
# covers rows PLACEMENT_ORIGIN + ci*PLACEMENT_TILE_SIZE .. +PLACEMENT_TILE_SIZE
# and the same for columns - the (s + ci*M, s + cj*M) convention of
# test_context_tiles.build_framed_map / mark_placement_cores and
# bruteforce.core_index. Every cell of a placement tile is a candidate (the
# placement/transit split: only a placement tile's own cells can ever be
# chosen by a placement decision); every other tile is a transit tile.
# Set it with Pyramid.set_placement_tiles (test_context_tiles.
# mark_placement_cores does so), read it with Pyramid.is_candidate.
PLACEMENT_TILES = frozenset()
PLACEMENT_TILE_SIZE = 3
PLACEMENT_ORIGIN = 0


class CoreItem:
    """elements: object array (base-level SquareItem or CoreItem, never
    mixed; 3x3 except at the far edge of the board, see Upstream.build).
    margin: the read-only items around them. realm: the rectangle of
    base-lattice ids this core owns. links: see the module docstring.
    """
    def __init__(self, elements, margin, realm):
        self.elements = elements
        self.margin = margin
        self.realm = realm
        self.forced = set()
        self.links = set()


class Pyramid:
    """The whole pyramid over one map: .base (Base, holding the map),
    .upstream (Upstream, holding the hierarchy as .pyramid after build),
    .downstream (Downstream). Also holds the functions both directions
    share, as static methods.

    map_: the map (object array of SquareItem); may be None for pyramids
    built by hand from synthetic CoreItems that never touch the map.
    tile_size, origin: base tiles are tile_size x tile_size, starting at
    row/col `origin`.
    """
    def __init__(self, map_, tile_size=3, origin=0):
        self.base = Base(map_, tile_size, origin)
        self.upstream = Upstream(self)
        self.downstream = Downstream(self)

    def onepass(self):
        """One up pass and one down pass over the whole map, in the order of
        the module docstring's "Flow". Returns base.map_onepass."""
        base_items = self.base.generate_core_items()
        self.upstream.build(base_items)
        self.downstream.run(self.upstream.pyramid)
        return self.base.map_onepass

    # --- shared ---------------------------------------------------------------

    @staticmethod
    def in_realm(id_, realm):
        """True if id_ (a (row, col) base-lattice index) falls inside realm -
        ((row_start, row_end), (col_start, col_end)), both ends exclusive on
        row_end/col_end, matching image_to_squares.core_range_for_tile.
        """
        (row_start, row_end), (col_start, col_end) = realm
        row, col = id_
        return row_start <= row < row_end and col_start <= col < col_end

    @staticmethod
    def set_placement_tiles(tiles, tile_size, origin=0):
        """Set the global placement-tile layout: `tiles` the (ci, cj) indices
        of the placement tiles, `tile_size` the base tile edge M, `origin`
        the grid's offset s. Replaces any earlier layout."""
        global PLACEMENT_TILES, PLACEMENT_TILE_SIZE, PLACEMENT_ORIGIN
        PLACEMENT_TILES = frozenset(tiles)
        PLACEMENT_TILE_SIZE = tile_size
        PLACEMENT_ORIGIN = origin

    @staticmethod
    def is_candidate(id_):
        """True if id_ lies in a placement tile, wherever it is on the map -
        inside the caller's own realm or not, even off the board (negative
        or too-large indices are simply in no placement tile). Pure
        arithmetic on id_ and the global layout, no map access: the same
        answer test_context_tiles.mark_placement_cores writes into
        .placement_status, available to code that only has an id (e.g. a
        parent looking at a child triple's id_in) - .state is not
        consulted, same as mark_placement_cores.
        """
        row, col = id_
        ci = (row - PLACEMENT_ORIGIN) // PLACEMENT_TILE_SIZE
        cj = (col - PLACEMENT_ORIGIN) // PLACEMENT_TILE_SIZE
        return (ci, cj) in PLACEMENT_TILES

    @staticmethod
    def owning_child(core, id_):
        """Index into core.elements of the child whose realm holds id_, or
        None if id_ is outside every child (or is DUMMY)."""
        if id_ == DUMMY:
            return None
        for idx in np.ndindex(core.elements.shape):
            if Pyramid.in_realm(id_, core.elements[idx].realm):
                return idx
        return None

    @staticmethod
    def child_exits(core):
        """A LOCAL view for a core whose children are CoreItems: {id_out:
        set of id_exit outside core's realm}, read off the children's .links
        triples. Exit links that land in a sibling are internal to core and
        dropped. Used by the up step (where a chain leaves core) and the
        down step's consume (where a chain from a sibling leaves core)."""
        exits = {}
        for child in core.elements.flat:
            for _a, b, c in child.links:
                if c != DUMMY and not Pyramid.in_realm(c, core.realm):
                    exits.setdefault(b, set()).add(c)
        return exits

    @staticmethod
    def candidate_links(core):
        """The part of core.links whose id_in is a candidate (is_candidate) -
        chains that a placement decision can trigger, as opposed to chains
        from an ordinary cell nothing may ever reach. Replaces the former
        .placement_reach field (removed 2026-11-07): computed on demand, so
        it always matches .links, including triples the down pass added."""
        return {(a, b, c) for a, b, c in core.links if Pyramid.is_candidate(a)}

    @staticmethod
    def fold_into(core, found, entered):
        """Last step of both consume halves (Base.consume_aliases_down,
        Downstream.consume_aliases_down): add each origin's exit triples to
        core.links, or (origin, DUMMY, DUMMY) if its chain entered core but
        reached no exit."""
        for origin, hits in found.items():
            if hits:
                core.links |= hits
            elif origin in entered:
                core.links.add((origin, DUMMY, DUMMY))


class Base:
    """The map and everything that works on SquareItems.

    .map: the map (object array of SquareItem). Prerequisites for the up and
    down steps, not done here: every cell's SquareItem.forced_by holds its
    current conditions and SquareItem.forces matches it
    (bruteforce.update_forces), and the global placement-tile layout is set
    (Pyramid.set_placement_tiles).

    A base tile is a CoreItem whose elements are a tile_size x tile_size
    slice of .map (tile / tiles / generate_core_items).

    .map_onepass: the base-level array of CoreItems after a down pass,
    written by Downstream.run (None until then). It is the same array as
    Upstream.pyramid[0], updated in place - its CoreItems' .links now hold
    the up-pass triples plus everything the down pass added.
    """
    def __init__(self, map_, tile_size=3, origin=0):
        self.map = map_
        self.tile_size = tile_size
        self.origin = origin
        self.map_onepass = None

    def generate_core_items(self):
        """The base level: every whole tile of .map as a CoreItem (tiles),
        with its .links computed (update_links). This is what Upstream.build
        takes as input."""
        grid = self.tiles()
        for tile in grid.flat:
            Base.update_links(tile)
        return grid

    def tile(self, ci, cj):
        """Base tile (ci, cj): a CoreItem over .map's cells origin + ci*tile_size
        .. +tile_size (rows) and the same for columns. Its .links is empty
        until update_links runs."""
        size = self.tile_size
        r0, c0 = self.origin + ci * size, self.origin + cj * size
        return CoreItem(self.map[r0:r0 + size, c0:c0 + size], margin=None,
                        realm=((r0, r0 + size), (c0, c0 + size)))

    def tiles(self):
        """Every whole tile of .map, as an object array of base CoreItems
        indexed (ci, cj); cells before `origin` or in a partial tile at the
        far edge belong to no tile. .links not yet computed."""
        rows, cols = self.map.shape
        n_r = max(0, (rows - self.origin) // self.tile_size)
        n_c = max(0, (cols - self.origin) // self.tile_size)
        if n_r == 0 or n_c == 0:
            raise ValueError(f"no whole {self.tile_size}x{self.tile_size} tile fits in a "
                             f"{rows}x{cols} map from origin {self.origin}")
        grid = np.empty((n_r, n_c), dtype=object)
        for ci in range(n_r):
            for cj in range(n_c):
                grid[ci, cj] = self.tile(ci, cj)
        return grid

    @staticmethod
    def elements_by_id(tile):
        """{absolute cell_id: SquareItem} for every one of a base tile's
        elements. Used by edges, which needs to look an id up by its
        absolute (row, col) position."""
        (row_start, _), (col_start, _) = tile.realm
        rows, cols = tile.elements.shape
        return {(row_start + li, col_start + lj): tile.elements[li, lj]
                for li in range(rows) for lj in range(cols)}

    @staticmethod
    def edges(tile):
        """The two kinds of single-link edges of a base tile (was
        CoreItem._base_edges), plus a placement_status lookup for every id
        touched. Shared by update_links and consume_aliases_down.

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
        cell it directly forces.
        """
        by_id = Base.elements_by_id(tile)
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
    def reach(origin, start_nodes, interior_ids, forward_edges, exit_edges):
        """Breadth-first walk from start_nodes through interior_ids-only
        waypoints (via forward_edges), recording (origin, node, exit) for
        every exit link of every visited node. Shared base-level walk for
        update_links and consume_aliases_down - see edges for what the
        three edge/lookup structures mean.

        "Last cell inside" convention (Ivo, 2026-11-07): id_out is the LAST
        cell INSIDE the tile that has a link pointing out, not the cell
        outside it links to. That cell is then exactly the id_in of the
        neighbouring tile's own triple for the same chain (its forcer from
        outside), so a parent can join the two directly. Where the link
        actually points is the triple's third item, id_exit - one triple
        per exit link of that cell. Replaces the earlier "first cell
        outside" id_out, which never matched the next tile's id_in - see
        test_upstream_forced_by.py and upstream-forced-by.html /
        upstream-forced-by-current.html (project root).
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
                for exit_id in exit_edges.get(node, ()):
                    found.add((origin, node, exit_id))
                for nxt in forward_edges.get(node, ()):
                    if nxt in interior_ids and nxt not in visited:
                        next_frontier.add(nxt)
            frontier = next_frontier
        return found

    @staticmethod
    def update_links(tile):
        """Recompute a base tile's .links (the up step at base level; the
        CoreItem-children half is Upstream.update_links) - "we need only one
        item to force another item."

        A breadth-first walk (reach) from (a) every external id with a
        direct entry edge into the tile - a chain passing through, "how far
        the chosen items on different tiles can travel" - and (b) every one
        of this tile's own candidate cells (is_candidate) - a path
        originating there, added 2026-10-23, "for the placement tiles we
        also add to the list paths that originate inside the tile": a
        placement tile's own cells are genuine future placement decisions,
        unlike a transit tile's, which are only ever relays. Records
        (origin, last cell inside, exit) for every exit link of every
        reached cell. A candidate whose own cell links straight out gives
        (candidate, candidate, exit).

        Ins that die (2026-11-07): a start whose chain forces at least one
        cell inside the tile but reaches no exit gets (start, DUMMY, DUMMY).
        A candidate that links to nothing gets nothing (nothing dies there),
        and a start with at least one exit gets no DUMMY even if another of
        its branches dies (Ivo, 2026-11-07: "partly dying starts get no
        dummy").
        """
        interior_ids, forward_edges, exit_edges, _status_of = Base.edges(tile)

        results = set()

        external_ins = {fid for fid in forward_edges if fid not in interior_ids}
        for id_in in external_ins:
            found = Base.reach(id_in, forward_edges[id_in], interior_ids, forward_edges, exit_edges)
            # an external in always forces a cell inside (that is what makes
            # it an in), so no exit means its chain died in here
            results |= found or {(id_in, DUMMY, DUMMY)}

        candidate_starts = {cid for cid in interior_ids if Pyramid.is_candidate(cid)}
        for cell_id in candidate_starts:
            found = Base.reach(cell_id, {cell_id}, interior_ids, forward_edges, exit_edges)
            if not found and forward_edges.get(cell_id, set()) & interior_ids:
                found = {(cell_id, DUMMY, DUMMY)}
            results |= found

        tile.links = results

    @staticmethod
    def consume_aliases_down(tile, aliases_down):
        """The consume step at base level (the CoreItem-children half is
        Downstream.consume_aliases_down): fold `aliases_down` - this tile's
        inbox, the set its parent's Downstream.update_aliases_down collected
        for it - into its .links. Does not modify the set it is given.

        For every (origin, b) in aliases_down, b is either one of this
        tile's own cells (the chain passes through it) or, under the "last
        cell inside" convention, the cell just outside that links in (the
        chain enters here) - in that case the walk starts from the cells it
        links to, the same way update_links starts from an external id_in.
        Adds (origin, last cell inside, exit) for every exit link reached -
        origin, not b, since b is only ever a relay waypoint.

        Ins that die (2026-11-07, same rule as update_links): an origin
        whose chain enters this tile but reaches no exit through any of its
        messages gets (origin, DUMMY, DUMMY).
        """
        if not aliases_down:
            return
        interior_ids, forward_edges, exit_edges, _status_of = Base.edges(tile)
        found = {}       # origin -> triples with a real exit, over all its messages
        entered = set()  # origins whose chain enters this tile at all
        for origin, b in aliases_down:
            start = {b} if b in interior_ids else forward_edges.get(b, set()) & interior_ids
            if start:
                entered.add(origin)
            found.setdefault(origin, set()).update(
                Base.reach(origin, start, interior_ids, forward_edges, exit_edges))
        Pyramid.fold_into(tile, found, entered)


class Upstream:
    """The up step for CoreItems whose children are CoreItems, and build(),
    the whole up pass. owner: the owning Pyramid. .pyramid: the CoreItem
    hierarchy built by build() - a list of object arrays of CoreItems, the
    base level first, the 1x1 top last (empty until build runs)."""
    def __init__(self, owner):
        self.owner = owner
        self.pyramid = []

    @staticmethod
    def update_links(core):
        """Recompute .links of a core whose children are CoreItems (the
        base-level half is Base.update_links): the children's .links,
        joined by compose_up (one child's id_out is the next child's id_in).

        Start points (Ivo, 2026-11-07): every child id_in outside this realm
        (a chain entering from outside), plus every child id_in that is a
        candidate (is_candidate) - wherever inside this realm it lies. An
        id_in inside this realm that is not a candidate is an ordinary cell
        of some child - a relay waypoint, not a start - and is dropped here
        (the reason for the 2026-10-25 require_external_origin fix: such an
        origin, once interior, is not a placement decision).
        """
        child_links = set()
        for child in core.elements.flat:
            child_links |= child.links
        origins = {a for a, _b, _c in child_links
                   if not Pyramid.in_realm(a, core.realm) or Pyramid.is_candidate(a)}
        core.links = Upstream.compose_up(child_links, Pyramid.child_exits(core), origins)

    @staticmethod
    def compose_up(child_links, exits, origins):
        """Compose children's (id_in, id_out, id_exit) triples into the
        bigger realm's own .links - the walk behind update_links.

        "Last cell inside" convention (2026-11-07, see Base.reach): a
        child's id_out is its own last cell, and that same cell is the id_in
        of the neighbouring child's triple for the same chain. So
        child_links, unioned across all children, chains directly: (a, b, _)
        in one child, (b, c, _) in the next. The earlier version (id_out =
        first cell outside) never found a continuation, because the next
        child's pair started one cell earlier - every chain was cut at the
        first border between two children
        (test_three_tile_straight_chain_joins_across_child_borders).

        exits: {id_out: set of id_exit outside this realm}, read off the
        children's triples (a local view, see Pyramid.child_exits).
        origins: the ids to start from; the caller decides.

        Each origin is walked breadth-first along child_links (id_in ->
        id_out); for every visited node with exit links out of this realm,
        (origin, node, exit) is recorded per exit - node is then this core's
        own last cell inside. A node whose links only point into a sibling
        continues via that sibling's triples starting at it; if the sibling
        has none, or only (node, DUMMY, DUMMY), the chain died inside it
        and that branch ends.

        Ins that die (2026-11-07): an origin whose walk records no exit at
        all - every branch died somewhere inside this realm - is recorded as
        (origin, DUMMY, DUMMY), so the in stays visible further up. Every
        origin here has at least one child triple (that is how it became an
        origin), so its chain always enters this realm. DUMMY itself is
        never walked on.
        """
        edges = {}
        for a, b, _c in child_links:
            edges.setdefault(a, set()).add(b)

        results = set()
        for origin in origins:
            found = set()
            frontier = {origin}
            visited = set()
            while frontier:
                next_frontier = set()
                for node in frontier:
                    if node in visited or node == DUMMY:
                        continue
                    visited.add(node)
                    for exit_id in exits.get(node, ()):
                        found.add((origin, node, exit_id))
                    for nxt in edges.get(node, ()):
                        if nxt not in visited:
                            next_frontier.add(nxt)
                frontier = next_frontier
            results |= found or {(origin, DUMMY, DUMMY)}
        return results

    def build(self, base_items):
        """The up pass over the whole board (added 2026-11-07, Ivo: "recall
        the upstream procedure starting with base level ... implement it").

        base_items: the base level, as Base.generate_core_items returns it -
        an object array of base CoreItems with .links already computed.

        1. Parent level: the CoreItems of the level below, grouped 3x3; at
           the far edge a group may be smaller (2 or 1 rows/cols) when the
           count is not a multiple of 3. A parent's realm is the bounding
           box of its children's realms. Upstream.update_links on each,
           children first.
        2. Repeat 1 until a single CoreItem remains.

        Stores the hierarchy in .pyramid - base_items first, the 1x1 top
        last - and returns it.
        """
        levels = [base_items]
        while levels[-1].shape != (1, 1):
            below = levels[-1]
            b_r, b_c = below.shape
            level = np.empty(((b_r + 2) // 3, (b_c + 2) // 3), dtype=object)
            for pi in range(level.shape[0]):
                for pj in range(level.shape[1]):
                    children = below[3 * pi:3 * pi + 3, 3 * pj:3 * pj + 3]
                    first, last = children[0, 0].realm, children[-1, -1].realm
                    realm = ((first[0][0], last[0][1]), (first[1][0], last[1][1]))
                    parent = CoreItem(children.copy(), margin=None, realm=realm)
                    Upstream.update_links(parent)
                    level[pi, pj] = parent
            levels.append(level)
        self.pyramid = levels
        return levels


class Downstream:
    """The down step for CoreItems whose children are CoreItems, and run(),
    the whole down pass. owner: the owning Pyramid. .pyramid: the hierarchy
    run() was given (Upstream.pyramid), updated in place."""
    def __init__(self, owner):
        self.owner = owner
        self.pyramid = []

    def run(self, pyramid):
        """The down pass over the whole hierarchy (2026-11-07): `pyramid` is
        Upstream.pyramid (base level first, top last), handed over once the
        up pass reached the top.

        From the top level down to level 1, for every CoreItem of the level,
        update_aliases_down(core, level, inherited): every candidate whose
        chain reaches into a child - the children's own candidates
        (siblings informing each other) and every candidate the core
        received from its own parent (`inherited`, including candidates
        outside the core, like Z - Ivo, 2026-11-07: "all candidates must be
        copied down, including those outside P") - is delivered to each
        child it enters or passes through, and each child consumes that
        into its own .links (at level 1 the children are base tiles:
        Base.consume_aliases_down).

        `inherited` for a core is the inbox its parent's call filled for it
        - kept here, level to level, as local variables only (the top level
        receives nothing).

        After level 1 has updated the base level, the base-level array is
        written back to the owning Pyramid's Base as .map_onepass (the same
        array object as pyramid[0], updated in place). Returns it.
        """
        self.pyramid = pyramid
        inherited = {}
        for level in range(len(pyramid) - 1, 0, -1):
            next_inherited = {}
            for core in pyramid[level].flat:
                inbox = Downstream.update_aliases_down(core, level, inherited.get(id(core), set()))
                for idx in np.ndindex(core.elements.shape):
                    next_inherited[id(core.elements[idx])] = inbox[idx]
            inherited = next_inherited
        self.owner.base.map_onepass = pyramid[0]
        return pyramid[0]

    @staticmethod
    def update_aliases_down(core, level, inherited=frozenset()):
        """The down step for one core (added 2026-10-25 - "we need to follow
        through links between child supertiles of the same level"): tells
        each child about every candidate whose chain reaches into it, which
        the child's own up step can never see, then lets each child consume
        that into its own .links.

        Two sources, both delivered by the same walk (_walk):
        - the children's own candidates - every child id_in that
          is_candidate says is one. Example (Ivo's own): on a supertile with
          a 3x3 grid of children, (1,0) a placement tile - is there a path
          from (1,0) reaching all the way into (1,2), passing through
          (1,1)? Neither (1,0) nor (1,2) could find it alone;
        - `inherited`: what this core received from its own parent in this
          down pass, (candidate, cell) messages - see push_down.

        level: core's own level in the pyramid (base tiles are level 0, so
        level >= 1 here). It decides which consume step the children get:
        Base.consume_aliases_down for level 1 (children are base tiles),
        Downstream.consume_aliases_down above that.

        The inbox is a LOCAL variable, `aliases_down`: an object array with
        the same shape as core.elements, one set per child, built fresh on
        every call (Ivo, 2026-11-07: "make .aliases_down a local variable, a
        3x3 array of set type"). Returns it, so the caller (run, or a test)
        can see who was told what, and hand each child its inbox as that
        child's `inherited` one level further down.
        """
        aliases_down = np.empty(core.elements.shape, dtype=object)
        for idx in np.ndindex(core.elements.shape):
            aliases_down[idx] = set()
        graph = Downstream._graph(core)

        _owners, edges, _exits = graph
        origins = {a for a in edges if a != DUMMY and Pyramid.is_candidate(a)}
        for origin in origins:
            Downstream._walk(core, graph, origin, origin, Pyramid.owning_child(core, origin), aliases_down)
        Downstream.push_down(core, inherited, aliases_down, graph)

        consume = Base.consume_aliases_down if level == 1 else Downstream.consume_aliases_down
        for idx in np.ndindex(core.elements.shape):
            consume(core.elements[idx], aliases_down[idx])
        return aliases_down

    @staticmethod
    def push_down(core, inherited, aliases_down, graph=None):
        """Copy down every candidate this core received from its parent
        (2026-11-07, Ivo: "all candidates must be copied down, including
        those outside P, like Z"; replaces the earlier push_down, which
        copied the core's own triples to the child holding their id_in - a
        child that, for a triple spanning several children, did not even
        contain its id_out).

        inherited: (candidate, cell) messages from the parent's down step -
        cell is either the cell just outside this core where the
        candidate's chain enters, or one of this core's cells it passes
        through. For each, walk from cell along the children's triples and
        put (candidate, node) into the inbox (`aliases_down`, filled in
        place) of every child the chain enters or passes through - see
        _walk. Each child then derives its own triples for the candidate in
        the consume step: (candidate, its last cell, exit), or (candidate,
        DUMMY, DUMMY) where the chain dies in it.
        """
        if graph is None:
            graph = Downstream._graph(core)
        for origin, cell in inherited:
            Downstream._walk(core, graph, origin, cell, Pyramid.owning_child(core, origin), aliases_down)

    @staticmethod
    def _graph(core):
        """The children's triples as a walkable graph (shared by
        update_aliases_down and push_down): owners {(id_in, id_out): set of
        child indices whose .links hold that link}, edges {id_in: set of
        id_out}, exits {id_out: set of id_exit}."""
        owners, edges, exits = {}, {}, {}
        for idx in np.ndindex(core.elements.shape):
            for a, b, c in core.elements[idx].links:
                owners.setdefault((a, b), set()).add(idx)
                edges.setdefault(a, set()).add(b)
                if c != DUMMY:
                    exits.setdefault(b, set()).add(c)
        return owners, edges, exits

    @staticmethod
    def _walk(core, graph, origin, start, src_child, aliases_down):
        """Walk from `start` along the children's triples and record
        (origin, node) in the inbox of every child, other than src_child
        (the origin's own child, None if the origin lies outside core),
        that the chain
        - passes through: node is that child's own cell (its last cell on
          the chain);
        - enters: node is the id_in of a triple of that child (2026-11-07 -
          also catches a chain that dies in the first child it enters,
          where node is no child's last cell), or an exit link of node
          points into that child.
        DUMMY is never walked on."""
        owners, edges, exits = graph
        frontier = {start}
        visited = set()
        while frontier:
            next_frontier = set()
            for node in frontier:
                if node in visited or node == DUMMY:
                    continue
                visited.add(node)
                told = set()
                owner = Pyramid.owning_child(core, node)
                if owner is not None:
                    told.add(owner)
                for target in exits.get(node, ()):
                    if Pyramid.in_realm(target, core.realm):
                        entered = Pyramid.owning_child(core, target)
                        if entered is not None:
                            told.add(entered)
                for nxt in edges.get(node, ()):
                    told |= owners[(node, nxt)]
                    if nxt not in visited:
                        next_frontier.add(nxt)
                for idx in told - {src_child}:
                    aliases_down[idx].add((origin, node))
            frontier = next_frontier

    @staticmethod
    def consume_aliases_down(core, aliases_down):
        """The consume step for a core whose children are CoreItems (the
        base-level half is Base.consume_aliases_down): fold `aliases_down` -
        this core's inbox, the set its parent's update_aliases_down
        collected for it - into its .links. Does not modify the set it is
        given.

        For every (origin, b) in aliases_down, walks forward from b along
        the children's .links, and adds (origin, node, exit) for every exit
        link of a reached node out of this core's realm. The exits come from
        the children's triples (local, see Pyramid.child_exits): a chain
        from a sibling can enter at an ordinary cell and reach a child
        id_out that no triple of this core ends at.

        Ins that die (2026-11-07, same rule as the up step): an origin whose
        chain enters this core but reaches no exit through any of its
        messages gets (origin, DUMMY, DUMMY).

        No-op if core.elements is None - only synthetic unit tests of the
        informing logic build such CoreItems; there is nothing to walk.
        """
        if not aliases_down or core.elements is None:
            return
        edges = {}
        for child in core.elements.flat:
            for a, b, _c in child.links:
                edges.setdefault(a, set()).add(b)
        leaves = Pyramid.child_exits(core)
        found = {}       # origin -> triples with a real exit, over all its messages
        entered = set()  # origins whose chain enters this core at all
        for origin, b in aliases_down:
            hits = found.setdefault(origin, set())
            if edges.get(b) or b in leaves:
                entered.add(origin)
            frontier = {b}
            visited = set()
            while frontier:
                next_frontier = set()
                for node in frontier:
                    if node in visited or node == DUMMY:
                        continue
                    visited.add(node)
                    for exit_id in leaves.get(node, ()):
                        hits.add((origin, node, exit_id))
                    for nxt in edges.get(node, ()):
                        if nxt not in visited:
                            next_frontier.add(nxt)
                frontier = next_frontier
        Pyramid.fold_into(core, found, entered)


# Module-level name kept for bruteforce.py / vacuum.py, which import it.
in_realm = Pyramid.in_realm
