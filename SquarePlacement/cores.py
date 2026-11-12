"""Hierarchical pyramid over the base lattice, for the fixed-superlattice
closure refactor (see do_closure_old's own docstring for the problems
motivating it).

A patch id is a cell's own (row, col) index in the base lattice.

Structure (2026-11-08, Ivo: "rewrite the new class structure towards later
CPU / GPU implementation ... move the map, the map_after and the .pyramid
list out of the classes ... to a new class DeviceMemory. The "main" of the
algo would be on CPU. Base, Upstream and Downstream function will later
become kernel calls ... each step uses only data from one level"):

- DeviceMemory: all data between steps - .map (the cells), .pyramid (a list
  of object arrays of CoreItems, index = level, 0 = base tiles), .inbox
  (the down pass's messages, laid out like .pyramid), .map_after (the base
  level after the down pass), and the tiling constants tile_size, origin.
- CoreItem: one node of the pyramid, data only - an array of elements
  (SquareItem for a base tile, CoreItems above that), a margin of read-only
  references around them, the rectangular realm ((row_start, row_end),
  (col_start, col_end)) of base-lattice ids it owns, and .links.
- Base, Upstream, Downstream: no state of their own. Each has steps that
  read one level from DeviceMemory and write the next level there (kernel
  calls later), plus the per-item functions the steps apply (a kernel's
  per-item work):
    Base.init_map(mem)            .map (.state)   -> .map (conditions)
    Base.suspend_contradictory(mem) .map          -> .suspended
    Base.build_level0(mem)        .map            -> .pyramid[0]
    Upstream.build_level(mem, k)  .pyramid[k-1]   -> .pyramid[k]
    Downstream.down_level(mem, k) .pyramid[k], .inbox[k]
                                  -> .pyramid[k-1] (.links updated), .inbox[k-1]
                                     (and .map_after when k == 1)
- Pyramid: the CPU main (onepass), which only sequences those steps, and
  the functions the steps share (static).

Flow (Pyramid.onepass; Ivo, 2026-11-08: "The main (CPU) sets only
map.state and moves map to DeviceMemory (of GPU). Then Base calls some init
function setting .condition etc (kernel call). Then main calls another Base
function converting .map to .pyramid[0] (next kernel call) Then Upstream
functions are called by main etc. until map_after is done"):

    mem = DeviceMemory(map_with_only_state, tile_size, origin)   # CPU
    Base.init_map(mem)             # .state -> .forced_by, .forces, .placement_status
    Base.suspend_contradictory(mem)  # .map -> .suspended (candidates set aside for this pass)
    Base.build_level0(mem)         # .map -> .pyramid[0]
    Upstream.build_level(mem, k)   # .pyramid[k-1] -> .pyramid[k], k = 1, 2, ... to the top
    Downstream.down_level(mem, k)  # k = top .. 1; k == 1 writes .map_after

All map preprocessing happens in Base.init_map; the CPU sets nothing but
.state (and the global placement-tile layout, a constant).

Functions that used to branch on "elements are SquareItem / CoreItem" are
two functions, one in Base and one in Upstream/Downstream; the caller picks
by level (level 0 = base tiles).

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

from map_of_squares import StateEnum


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

# Candidates suspended for the current pass (2026-11-12) - cells of a
# placement tile that Pyramid.is_candidate must not count as candidates,
# because their own chain forces two cells that block each other (see
# Base.suspend_contradictory). Mirrors DeviceMemory.suspended for the pass, so
# is_candidate needs no DeviceMemory argument (like the layout above).
SUSPENDED = frozenset()


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


class DeviceMemory:
    """All data the steps read and write (2026-11-08) - what later lives in
    GPU memory, while the CPU main (Pyramid.onepass) only sequences steps.

    .map: the cells (object array of SquareItem), put here by the CPU. For
      now with their conditions (.forced_by, .forces) already computed -
      see the module docstring.
    .tile_size, .origin: base tiles are tile_size x tile_size cells of .map,
      starting at row/col origin (constants for the Base step).
    .pyramid: list of object arrays of CoreItems, index = level - [0] the
      base tiles (Base.build_level0), [k] the parents of [k-1]
      (Upstream.build_level), the last one 1x1 (the top).
    .inbox: list laid out like .pyramid - .inbox[k][i, j] is the set of
      (candidate, cell) messages the down step of CoreItem
      .pyramid[k][i, j]'s parent left for it (empty for the top). Written
      by Downstream.down_level(mem, k+1), read by down_level(mem, k). Was
      a local variable carried between levels before 2026-11-08; between
      two separate steps it has to live here.
    .suspended: the candidates suspended for this pass (2026-11-12) - see
      Base.suspend_contradictory; empty until that step runs.
    .map_after: the base level after the down pass - the same array as
      .pyramid[0], its CoreItems' .links holding the up-pass triples plus
      everything the down pass added (None until Downstream reaches level
      1; was Base.map_onepass).
    """
    def __init__(self, map_, tile_size=3, origin=0):
        self.map = map_
        self.tile_size = tile_size
        self.origin = origin
        self.pyramid = []
        self.inbox = []
        self.map_after = None
        self.suspended = set()


class Pyramid:
    """The CPU main (onepass) and the functions the steps share, as static
    methods. Holds no data - everything lives in DeviceMemory."""

    @staticmethod
    def upward(mem):
        """CPU main, up half: Base.build_level0, then for level 1, 2, ...
        until a level holds a single CoreItem: Upstream.build_level, followed
        by the round trip from that level (option 2, see round_trip).
        Returns mem.pyramid."""
        Base.build_level0(mem)
        while mem.pyramid[-1].shape != (1, 1):
            level = len(mem.pyramid)
            Upstream.build_level(mem, level)
            Pyramid.round_trips(mem, level, Pyramid.empty_inbox(mem.pyramid[level].shape))
        return mem.pyramid

    @staticmethod
    def downward(mem):
        """CPU main, down half: an empty inbox for the top, then from the
        top level down to level 1: Downstream.down_level (which updates the
        level below and writes its inbox; at level 1 also mem.map_after),
        followed by the round trip from the level just updated (option 2,
        see round_trip). Returns mem.map_after."""
        top = len(mem.pyramid) - 1
        mem.inbox = [None] * len(mem.pyramid)
        mem.inbox[top] = Pyramid.empty_inbox(mem.pyramid[top].shape)
        for level in range(top, 0, -1):
            Downstream.down_level(mem, level)
            Pyramid.round_trips(mem, level - 1, mem.inbox[level - 1])
        return mem.map_after

    @staticmethod
    def round_trips(mem, level, inbox):
        """Repeat round_trip until it changes nothing (n-ary generation only
        ever adds conditions, so this terminates). Returns the number of
        round trips that changed something."""
        changed = 0
        while Pyramid.round_trip(mem, level, inbox):
            changed += 1
        return changed

    @staticmethod
    def round_trip(mem, level, inbox):
        """Option 2 of TODO item 5 (Ivo, 2026-11-08/10: "each time links of a
        supertile are generated, we go down the pyramid to base level,
        generate links by n-ary ..., go up to that level. Downstream, we do
        the same."). Called by the CPU main right after the step that
        generated mem.pyramid[level]'s links - after Upstream.build_level on
        the way up, after Downstream.down_level (which updated this level)
        on the way down.

        inbox: what each CoreItem of the level knows about candidates from
        above, laid out like mem.pyramid[level] - empty sets on the way up
        (nothing from above yet), mem.inbox[level] on the way down.

        1. Down (Downstream.inform_level, level by level to 1): each
           CoreItem tells its children which candidates' chains enter or
           pass through them - the down pass's message walk, but read-only
           (nothing is consumed into .links). At base level every tile then
           has its simultaneous entries.
        2. At base level (Base.nary_level): generate links from n-ary
           conditions out of each tile's simultaneous entries - writes new
           conditions into mem.map, reports which tiles changed. A stub for
           now (Base.nary_links changes nothing).
        3. Up (Base.relink_level, then Upstream.relink_level to `level`):
           recompute the changed tiles and their ancestors, ADDING the
           result to their .links - only additions ever happen in a pass,
           and replacing would erase what the down pass already delivered.

        Returns True if step 2 changed anything (the caller repeats until
        it does not).
        """
        entries = inbox
        for j in range(level, 0, -1):
            entries = Downstream.inform_level(mem, j, entries)
        changed = Base.nary_level(mem, entries)
        if not changed.any():
            return False
        Base.relink_level(mem, changed)
        for j in range(1, level + 1):
            changed = Upstream.relink_level(mem, j, changed)
        return True

    @staticmethod
    def onepass(mem):
        """The CPU main (2026-11-08): the CPU has put a map with only .state
        set into mem.map; Base.init_map derives the conditions from it, then
        one up pass and one down pass follow - a sequence of steps that each
        read one level of mem and write the next. Returns mem.map_after.

        (upward/downward alone skip init_map - for a map whose conditions
        are already in place, e.g. hand-built in a test.)"""
        Base.init_map(mem)
        Base.suspend_contradictory(mem)
        Pyramid.upward(mem)
        return Pyramid.downward(mem)

    @staticmethod
    def empty_inbox(shape):
        """An object array of empty sets - one (candidate, cell) inbox per
        CoreItem of a level (see DeviceMemory.inbox)."""
        inbox = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            inbox[idx] = set()
        return inbox

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

        A candidate suspended for the current pass (2026-11-12, see
        Base.suspend_contradictory) is not a candidate.
        """
        return Pyramid.in_placement_tile(id_) and id_ not in SUSPENDED

    @staticmethod
    def in_placement_tile(id_):
        """True if id_ lies in a placement tile (the global layout only) -
        is_candidate without the suspension of Base.suspend_contradictory.
        This is what .placement_status and the same-tile pruning use: one
        chosen cell per placement tile is a fixed, structural fact."""
        row, col = id_
        ci = (row - PLACEMENT_ORIGIN) // PLACEMENT_TILE_SIZE
        cj = (col - PLACEMENT_ORIGIN) // PLACEMENT_TILE_SIZE
        return (ci, cj) in PLACEMENT_TILES

    @staticmethod
    def set_suspended(ids):
        """Set the candidates suspended for the current pass (see SUSPENDED,
        Base.suspend_contradictory)."""
        global SUSPENDED
        SUSPENDED = frozenset(ids)

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
    def add_links(core, new):
        """Add `new` triples to core.links, keeping the "partly dying starts
        get no dummy" rule (Ivo, 2026-11-07) across the union: a start that
        has a real triple afterwards loses any (start, DUMMY, DUMMY) - its
        chain gained an exit since (e.g. through a link n-ary generation
        added). Used wherever links are added rather than recomputed."""
        links = core.links | new
        exiting = {a for a, b, _c in links if b != DUMMY}
        core.links = {t for t in links if not (t[1] == DUMMY and t[0] in exiting)}

    @staticmethod
    def fold_into(core, found, entered):
        """Last step of both consume halves (Base.consume_aliases_down,
        Downstream.consume_aliases_down): add each origin's exit triples to
        core.links, or (origin, DUMMY, DUMMY) if its chain entered core but
        reached no exit."""
        new = set()
        for origin, hits in found.items():
            if hits:
                new |= hits
            elif origin in entered:
                new.add((origin, DUMMY, DUMMY))
        Pyramid.add_links(core, new)


class Base:
    """Everything that works on SquareItems - stateless; the cells are in
    DeviceMemory.map.

    Prerequisite, set by the CPU: the global placement-tile layout
    (Pyramid.set_placement_tiles). Everything else about the cells beyond
    .state is derived here, by init_map.

    A base tile is a CoreItem whose elements are a tile_size x tile_size
    slice of mem.map (tile / tiles).
    """

    @staticmethod
    def init_map(mem):
        """The Base init step (a kernel later; 2026-11-08, Ivo: "move all map
        preprocessing to Base. The main (CPU) sets only map.state and moves
        map to DeviceMemory ... Then Base calls some init function setting
        .condition etc"): reads each cell's .state in mem.map and the global
        placement-tile layout, and writes the cell's .forced_by (its
        conditions), .forces (their reverse index) and .placement_status, in
        place. Whatever those fields held before is overwritten.

        1. Vacuum conditions from the board's geometry alone
           (conditions.initial_map): every triple whose three placements lie
           on the board, members tagged with their placement_status.
        1a. Already blocked corners (2026-11-12, Ivo: "there must be a seat
           at (9,5)"): the vacuum assumes no corner is blocked yet, and
           step 3 only reduces by the members' own states - a corner that is
           blocked without any chosen cell behind it (the frozen area's
           border, set blocked directly) would still demand a blocker. So
           every free cell also gets blocked_corner_conditions: per 2x2
           block, a blocker only for the corners still open.
        2. .placement_status from the layout (is_candidate: every cell of a
           placement tile is a candidate, every other cell unknown), and
           the condition tags refreshed to match (bruteforce.retag_forced_by)
           - what test_context_tiles.mark_placement_cores + retag_forced_by
           did on the CPU before.
        3. Reduce by .state (bruteforce.compute_conditions): drop every
           condition with a blocked member, remove chosen members, clear the
           conditions of non-free cells. States only ever go free -> chosen
           or free -> blocked, so one reduction with the final states gives
           what reducing after every placement phase gave.
        4. Drop conditions naming two candidates of the same placement tile
           (bruteforce.prune_same_core_candidates) - a placement tile only
           ever has one of its own cells chosen.

        .forces is kept consistent with .forced_by throughout (each of the
        bruteforce functions ends with update_forces).
        """
        # imported here: bruteforce imports cores (in_realm), so importing it
        # at module level would be circular
        from conditions import initial_map
        from bruteforce import retag_forced_by, compute_conditions, prune_same_core_candidates
        from map_of_squares import PlacementStatus

        m = mem.map
        mem.suspended = set()
        Pyramid.set_suspended(())
        vacuum = initial_map(m.shape)
        for idx in np.ndindex(m.shape):
            cell = m[idx]
            cell.forced_by = vacuum[idx].forced_by
            cell.forces = vacuum[idx].forces
            cell.placement_status = (PlacementStatus.candidate if Pyramid.in_placement_tile(idx)
                                     else PlacementStatus.unknown)
        for idx in np.ndindex(m.shape):
            if m[idx].state == StateEnum.free:
                m[idx].forced_by |= {frozenset((pos, PlacementStatus.unknown) for pos in blockers)
                                     for blockers in Base.blocked_corner_conditions(m, idx)}
        retag_forced_by(m)
        compute_conditions(m)
        prune_same_core_candidates(m, mem.tile_size, mem.origin)

    @staticmethod
    def blocked_corner_conditions(m, x):
        """The conditions of cell x that take the corners already blocked in
        m into account (2026-11-12; see init_map, step 1a): for each of the
        four 2x2 blocks around x - x and its three other corners - x is a
        seat once all three are blocked. A corner already blocked needs
        nothing; each corner still open needs one chosen diagonal neighbour
        of its own. Blockers follow the vacuum's rules
        (conditions.vacuum_condition_offsets): never x, a corner of this
        block, or a diagonal neighbour of x (that would block x itself);
        never two blockers diagonal to each other; only free, on-board
        cells. A block with a corner off the board, or a chosen corner, is
        skipped (it can never become a seat that way).

        Returns a set of frozensets of blocker ids (plain ids, untagged) -
        a single blocker where two corners are already blocked, e.g.
        {(8,7)} for (9,5) in the frozen area, whose lower corners are the
        blocked border. A block with all three corners already blocked
        would make x a seat right now: there is no blocker to name, so it
        yields nothing here (the CPU places seats)."""
        from itertools import combinations, product
        rows, cols = m.shape
        diagonals = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        out = set()
        for dr, dc in diagonals:
            corners = [(x[0] + dr, x[1]), (x[0], x[1] + dc), (x[0] + dr, x[1] + dc)]
            if any(not (0 <= r < rows and 0 <= c < cols) for r, c in corners):
                continue
            if any(m[k].state == StateEnum.chosen for k in corners):
                continue
            open_corners = [k for k in corners if m[k].state != StateEnum.blocked]
            if not open_corners:
                continue
            forbidden = set(corners) | {x} | {(x[0] + a, x[1] + b) for a, b in diagonals}
            per_corner = [[(k[0] + a, k[1] + b) for a, b in diagonals
                           if (k[0] + a, k[1] + b) not in forbidden
                           and 0 <= k[0] + a < rows and 0 <= k[1] + b < cols
                           and m[k[0] + a, k[1] + b].state == StateEnum.free]
                          for k in open_corners]
            for blockers in product(*per_corner):
                if any(abs(p[0] - q[0]) == 1 and abs(p[1] - q[1]) == 1
                       for p, q in combinations(blockers, 2)):
                    continue
                out.add(frozenset(blockers))
        return out

    @staticmethod
    def suspend_contradictory(mem):
        """The Base suspension step (a kernel later; 2026-11-12, Ivo: "If a
        cell A forces two cells that block each other, deprive A of its
        candidate status. This is a harmless postponement. Once one of the
        blocked cells is chosen, A becomes eligible again."): reads mem.map
        (after init_map), writes mem.suspended.

        For every free cell A of a placement tile: its chain - every cell A
        forces through single links (size-1 conditions), map-wide. If two
        cells of A's chain plus A itself are diagonal neighbours - they block
        each other, so A can never validly be chosen - A is suspended: for
        the rest of this pass it is no candidate (Pyramid.is_candidate), so
        it starts no chain, is no start point at any level, and the n-ary
        step does not use it. Known cases: (3,5) on the swept map with seed
        0 (its chain reaches its own diagonal neighbours (2,4), (2,6)), and
        (16,17) / (16,15) with seed 2 (forced seats diagonal to each other).

        Not sticky: init_map clears it, and this step derives it again from
        the current state on every pass - once one of the conflicting cells
        is chosen, the contradiction is gone and A is a candidate again.
        Checked once per pass, before the n-ary step: a candidate that only
        becomes contradictory through links the n-ary step adds is not
        caught yet.

        Returns the suspended set.
        """
        m = mem.map
        forcers = {}
        for idx in np.ndindex(m.shape):
            for condition in m[idx].forced_by:
                if len(condition) == 1:
                    (a, _s), = condition
                    forcers.setdefault(a, set()).add(idx)
        suspended = set()
        for a in np.ndindex(m.shape):
            if m[a].state != StateEnum.free or not Pyramid.in_placement_tile(a):
                continue
            chain, frontier = {a}, [a]
            while frontier:
                x = frontier.pop()
                for y in forcers.get(x, ()):
                    if y not in chain:
                        chain.add(y)
                        frontier.append(y)
            if any((x[0] + dr, x[1] + dc) in chain
                   for x in chain for dr, dc in ((1, -1), (1, 1))):
                suspended.add(a)
        mem.suspended = suspended
        Pyramid.set_suspended(suspended)
        return suspended

    @staticmethod
    def build_level0(mem):
        """The Base step (a kernel later): reads mem.map, writes
        mem.pyramid[0] - every whole tile of the map as a CoreItem (tiles),
        with its .links computed (update_links). Starts a new pyramid:
        any earlier levels are dropped."""
        grid = Base.tiles(mem)
        for tile in grid.flat:
            Base.update_links(tile)
        mem.pyramid = [grid]
        return grid

    @staticmethod
    def tile(mem, ci, cj):
        """Base tile (ci, cj): a CoreItem over mem.map's cells origin +
        ci*tile_size .. +tile_size (rows) and the same for columns. Its
        .links is empty until update_links runs."""
        size = mem.tile_size
        r0, c0 = mem.origin + ci * size, mem.origin + cj * size
        return CoreItem(mem.map[r0:r0 + size, c0:c0 + size], margin=None,
                        realm=((r0, r0 + size), (c0, c0 + size)))

    @staticmethod
    def tiles(mem):
        """Every whole tile of mem.map, as an object array of base CoreItems
        indexed (ci, cj); cells before `origin` or in a partial tile at the
        far edge belong to no tile. .links not yet computed."""
        rows, cols = mem.map.shape
        n_r = max(0, (rows - mem.origin) // mem.tile_size)
        n_c = max(0, (cols - mem.origin) // mem.tile_size)
        if n_r == 0 or n_c == 0:
            raise ValueError(f"no whole {mem.tile_size}x{mem.tile_size} tile fits in a "
                             f"{rows}x{cols} map from origin {mem.origin}")
        grid = np.empty((n_r, n_c), dtype=object)
        for ci in range(n_r):
            for cj in range(n_c):
                grid[ci, cj] = Base.tile(mem, ci, cj)
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
    def compute_links(tile):
        """A base tile's links, computed from its cells and returned (not
        stored - update_links stores them; the CoreItem-children half is
        Upstream.compute_links) - "we need only one item to force another
        item."

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

        A cell suspended for this pass (Base.suspend_contradictory) is no
        start at all, not even as an external id linking in (2026-11-12, Ivo:
        "do it" - a suspended cell leaves no trace in the pass). That cuts
        no chain that is not contradictory anyway: a chain that runs through
        a suspended cell contains that cell's own conflicting pair, so its
        start is suspended too.

        Ins that die (2026-11-07): a start whose chain forces at least one
        cell inside the tile but reaches no exit gets (start, DUMMY, DUMMY).
        A candidate that links to nothing gets nothing (nothing dies there),
        and a start with at least one exit gets no DUMMY even if another of
        its branches dies (Ivo, 2026-11-07: "partly dying starts get no
        dummy").
        """
        interior_ids, forward_edges, exit_edges, _status_of = Base.edges(tile)

        results = set()

        external_ins = {fid for fid in forward_edges
                        if fid not in interior_ids and fid not in SUSPENDED}
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

        return results

    @staticmethod
    def update_links(tile):
        """Set a base tile's .links to compute_links(tile)."""
        tile.links = Base.compute_links(tile)

    @staticmethod
    def nary_level(mem, entries):
        """The n-ary step at base level (a kernel later; option 2, see
        Pyramid.round_trip): for every base tile, nary_links with that
        tile's simultaneous entries. entries: laid out like mem.pyramid[0],
        each a set of (candidate, cell) messages.

        nary_links reports the CELLS whose conditions it changed - they can
        lie in other tiles than the one whose entries produced them (an
        alias found in tile (1,1) can collapse a condition of a cell in tile
        (2,2)). Each changed cell marks the base tile that holds it. Keeps
        .forces consistent (update_forces) if anything changed. Returns a
        boolean array laid out like mem.pyramid[0] - True for every tile
        holding a changed cell."""
        tiles = mem.pyramid[0]
        changed_cells = set()
        for idx, tile in np.ndenumerate(tiles):
            changed_cells |= set(Base.nary_links(mem, tile, entries[idx]))
        changed = np.zeros(tiles.shape, dtype=bool)
        size, origin = mem.tile_size, mem.origin
        for r, c in changed_cells:
            ci, cj = (r - origin) // size, (c - origin) // size
            if 0 <= ci < tiles.shape[0] and 0 <= cj < tiles.shape[1]:
                changed[ci, cj] = True
        if changed_cells:
            from bruteforce import update_forces   # bruteforce imports cores: import here
            update_forces(mem.map)
        return changed

    @staticmethod
    def nary_links(mem, tile, entries):
        """Generate links from n-ary conditions for one base tile - first
        version, single candidates only (2026-11-10, Ivo: "candidate (6,7)
        forces (6,6). one level up (6,6) gets alias (6,7). This affects all
        .conditions, including tile (2,2). in cell (8,7) a term collapses,
        creating a link.").

        `entries`: the tile's simultaneous entries, (candidate, cell)
        messages from the round trip's descent - cell is where the
        candidate's chain enters the tile (the cell just outside, linking
        in) or one of the tile's own cells it passes through. The tile's
        own free candidates count as entries (c, c) as well.

        For every candidate c among them:
        1. Aliases of c: c itself, its entry cells, and every cell of this
           tile its chain reaches from there along single links (Base.edges)
           - whenever c is chosen, all of them are.
        2. Collapse, for every free cell y (anywhere on the map - conditions
           reach up to two cells, so y can lie in another tile) and each of
           its pair or triple conditions, all aliases of c merge into c:
           - ALL members aliases: the condition collapses to c alone - y
             gets the single link c -> y;
           - two or more members aliases, but not all (2026-11-12, Ivo:
             "partial collapses should be present in .condition data"): y
             gets the reduced condition {c} + the non-alias members - a
             triple with two aliases becomes a pair with c;
           - exactly one member an alias: replacing it by c would not reduce
             the condition, only rename a member - left out.
           Size-1 conditions are left alone - they are links already, and
           collapsing them would only add shortcuts along existing chains.
           The original conditions stay - c forcing an alias does not rule
           out that alias becoming chosen some other way.
        Compounds of several candidates completing a condition together are
        not done yet.

        Writes the new conditions into mem.map (c tagged with its
        placement_status, the other members keep their tags) and returns the
        set of cells it changed (empty if none): every y that got a new
        condition, and c itself when y got a link - c's reverse index
        (.forces) gains y, which can be a new exit link out of c's own tile
        (a reduced pair or triple never is: only size-1 conditions make
        links).
        nary_level keeps .forces consistent and marks the tiles.
        """
        m = mem.map
        rows, cols = m.shape
        interior_ids, forward_edges, _exit_edges, _status_of = Base.edges(tile)
        by_candidate = {}
        for candidate, cell in entries:
            by_candidate.setdefault(candidate, set()).add(cell)
        for cid in interior_ids:
            if Pyramid.is_candidate(cid) and m[cid].state == StateEnum.free:
                by_candidate.setdefault(cid, set()).add(cid)

        changed = set()
        for candidate, cells in by_candidate.items():
            aliases = {candidate}
            frontier = set()
            for cell in cells:
                aliases.add(cell)
                frontier |= {cell} if cell in interior_ids else forward_edges.get(cell, set()) & interior_ids
            while frontier:
                node = frontier.pop()
                if node in aliases and node not in cells:
                    continue
                aliases.add(node)
                frontier |= (forward_edges.get(node, set()) & interior_ids) - aliases
            link = frozenset({(candidate, m[candidate].placement_status)})
            on_board = [a for a in aliases if 0 <= a[0] < rows and 0 <= a[1] < cols]
            r0 = max(0, min(a[0] for a in on_board) - 2)
            r1 = min(rows, max(a[0] for a in on_board) + 3)
            c0 = max(0, min(a[1] for a in on_board) - 2)
            c1 = min(cols, max(a[1] for a in on_board) + 3)
            c_member = (candidate, m[candidate].placement_status)
            for y in np.ndindex(r1 - r0, c1 - c0):
                y = (y[0] + r0, y[1] + c0)
                cell_y = m[y]
                if y == candidate or cell_y.state != StateEnum.free or link in cell_y.forced_by:
                    continue
                added = set()
                for condition in cell_y.forced_by:
                    if len(condition) < 2:
                        continue
                    others = frozenset(mem_ for mem_ in condition if mem_[0] not in aliases)
                    n_alias = len(condition) - len(others)
                    if n_alias == len(condition):
                        added = {link}          # full collapse: a link, nothing else needed
                        break
                    if n_alias >= 2:            # partial collapse: fewer terms
                        added.add(others | {c_member})
                added -= cell_y.forced_by
                if added:
                    cell_y.forced_by |= added
                    changed.add(y)
                    if link in added:
                        changed.add(candidate)
        return changed

    @staticmethod
    def relink_level(mem, changed):
        """Up, at base level (option 2, see Pyramid.round_trip): for every
        base tile marked in `changed` (laid out like mem.pyramid[0]),
        recompute its links from its cells and ADD them to .links."""
        for idx, tile in np.ndenumerate(mem.pyramid[0]):
            if changed[idx]:
                Pyramid.add_links(tile, Base.compute_links(tile))

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
    """The up step - stateless; reads and writes DeviceMemory.pyramid."""

    @staticmethod
    def update_links(core):
        """Set .links of a core whose children are CoreItems to
        compute_links(core)."""
        core.links = Upstream.compute_links(core)

    @staticmethod
    def relink_level(mem, level, changed_below):
        """Up, one level (option 2, see Pyramid.round_trip): every CoreItem
        of mem.pyramid[level] with a child marked in `changed_below` (laid
        out like mem.pyramid[level-1]) recomputes its links from its
        children and ADDS them to .links. Returns the marks for this level,
        for the next one up."""
        parents = mem.pyramid[level]
        changed = np.zeros(parents.shape, dtype=bool)
        for (pi, pj), core in np.ndenumerate(parents):
            if changed_below[3 * pi:3 * pi + 3, 3 * pj:3 * pj + 3].any():
                Pyramid.add_links(core, Upstream.compute_links(core))
                changed[pi, pj] = True
        return changed

    @staticmethod
    def compute_links(core):
        """The links of a core whose children are CoreItems, computed and
        returned (not stored - update_links stores them; the base-level half
        is Base.compute_links): the children's .links, joined by compose_up
        (one child's id_out is the next child's id_in).

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
        return Upstream.compose_up(child_links, Pyramid.child_exits(core), origins)

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

    @staticmethod
    def build_level(mem, level):
        """The Upstream step (a kernel later): reads mem.pyramid[level-1],
        writes mem.pyramid[level] - the CoreItems of the level below,
        grouped 3x3 (at the far edge a group may be smaller, 2 or 1
        rows/cols, when the count is not a multiple of 3), each with its
        .links computed (update_links). A parent's realm is the bounding box
        of its children's realms. level must be len(mem.pyramid) (the next
        level to build)."""
        assert level == len(mem.pyramid), (level, len(mem.pyramid))
        below = mem.pyramid[level - 1]
        b_r, b_c = below.shape
        grid = np.empty(((b_r + 2) // 3, (b_c + 2) // 3), dtype=object)
        for pi in range(grid.shape[0]):
            for pj in range(grid.shape[1]):
                children = below[3 * pi:3 * pi + 3, 3 * pj:3 * pj + 3]
                first, last = children[0, 0].realm, children[-1, -1].realm
                realm = ((first[0][0], last[0][1]), (first[1][0], last[1][1]))
                parent = CoreItem(children.copy(), margin=None, realm=realm)
                Upstream.update_links(parent)
                grid[pi, pj] = parent
        mem.pyramid.append(grid)
        return grid


class Downstream:
    """The down step - stateless; reads and writes DeviceMemory.pyramid and
    DeviceMemory.inbox."""

    @staticmethod
    def down_level(mem, level):
        """The Downstream step for one level (a kernel later): reads
        mem.pyramid[level] and mem.inbox[level], updates the .links of
        mem.pyramid[level-1] and writes mem.inbox[level-1].

        For every CoreItem of the level, update_aliases_down(core, level,
        inherited) with inherited = its own inbox: every candidate whose
        chain reaches into a child - the children's own candidates
        (siblings informing each other) and every candidate the core
        received from its own parent, including candidates outside the core,
        like Z (Ivo, 2026-11-07: "all candidates must be copied down,
        including those outside P") - is delivered to each child it enters
        or passes through, and each child consumes that into its own .links
        (at level 1 the children are base tiles: Base.consume_aliases_down).
        The inbox each child was given goes to mem.inbox[level-1] at the
        child's own grid position (parent (pi, pj) has children (3*pi + i,
        3*pj + j)).

        At level 1 the base level is final: mem.map_after = mem.pyramid[0]
        (Ivo, 2026-11-07: "After updating the lowest CoreItem level
        Downstream writes the lowest-level array back").
        """
        parents = mem.pyramid[level]
        children = mem.pyramid[level - 1]
        child_inbox = Pyramid.empty_inbox(children.shape)
        for (pi, pj), core in np.ndenumerate(parents):
            told = Downstream.update_aliases_down(core, level, mem.inbox[level][pi, pj])
            for (i, j), messages in np.ndenumerate(told):
                child_inbox[3 * pi + i, 3 * pj + j] = messages
        mem.inbox[level - 1] = child_inbox
        if level == 1:
            mem.map_after = mem.pyramid[0]
        return child_inbox

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
        3x3 array of set type"). Returns it, so the caller (down_level, or a
        test) can see who was told what; down_level stores it in
        DeviceMemory.inbox as each child's `inherited` one level further
        down.
        """
        aliases_down = Downstream.inform(core, inherited)
        consume = Base.consume_aliases_down if level == 1 else Downstream.consume_aliases_down
        for idx in np.ndindex(core.elements.shape):
            consume(core.elements[idx], aliases_down[idx])
        return aliases_down

    @staticmethod
    def inform(core, inherited=frozenset()):
        """The message half of update_aliases_down, without the consume
        step: returns the local inbox array (shape of core.elements, one set
        per child) of (candidate, cell) messages - from the children's own
        candidates and from `inherited` (push_down). Reads only; changes
        nothing. Used directly by the read-only descent of
        Pyramid.round_trip (inform_level)."""
        aliases_down = Pyramid.empty_inbox(core.elements.shape)
        graph = Downstream._graph(core)
        _owners, edges, _exits = graph
        origins = {a for a in edges if a != DUMMY and Pyramid.is_candidate(a)}
        for origin in origins:
            Downstream._walk(core, graph, origin, origin, Pyramid.owning_child(core, origin), aliases_down)
        Downstream.push_down(core, inherited, aliases_down, graph)
        return aliases_down

    @staticmethod
    def inform_level(mem, level, inbox):
        """Down one level, read-only (option 2, see Pyramid.round_trip):
        inform for every CoreItem of mem.pyramid[level] with its own inbox
        (`inbox`, laid out like mem.pyramid[level]). Returns the children's
        inboxes, laid out like mem.pyramid[level-1] (parent (pi, pj) has
        children (3*pi + i, 3*pj + j)). Writes nothing to mem."""
        child_inbox = Pyramid.empty_inbox(mem.pyramid[level - 1].shape)
        for (pi, pj), core in np.ndenumerate(mem.pyramid[level]):
            for (i, j), messages in np.ndenumerate(Downstream.inform(core, inbox[pi, pj])):
                child_inbox[3 * pi + i, 3 * pj + j] = messages
        return child_inbox

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
