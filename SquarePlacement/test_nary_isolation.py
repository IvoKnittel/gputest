"""N-ary link generation in isolation (2026-11-10, Ivo: "First, we test
n-ary link generation in isolation. For this we use 1. The maps from your
point 1 with 3x3 SquareItem cores ... (one map with different random
numbers, that is) ... I will place more cells there by hand later. 2. the
map from test_frozen_area.").

Each test:
1. builds a map with only .state set (what the CPU main hands over) and
   sets the placement-tile layout;
2. runs the CPU main (Pyramid.onepass) with Base.nary_links wrapped by a
   recorder, which collects every base tile's simultaneous entries over all
   round trips (up and down) and then runs the real step;
3. calls Base.nary_links for every base tile in isolation, with those
   entries - the step under test. The round trips have already run it to a
   fixed point, so it must find nothing new;
4. writes the map, its states, its links and every tile's entries as an
   HTML page (links_html.render_links_html) into html_out/ in the project
   root, for looking at the scenarios.
"""
import os

import numpy as np

from closure import do_closure
from cores import DeviceMemory, Pyramid, Base, DUMMY
from links_html import (render_links_html, render_candidate_maps_html,
                        record_candidate_steps, render_candidate_steps_html, candidate_schema)
from map_of_squares import SquareItem, StateEnum
from representation import build_margin_free_map, build_zipper_margin_map
from test_context_tiles import build_swept_map, placement_core_indices

HTML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "html_out")
SWEPT_SEEDS = (0, 1, 2)


def _state_only(m):
    """A fresh map with nothing set but .state, copied from m."""
    fresh = np.empty(m.shape, dtype=object)
    for idx in np.ndindex(m.shape):
        fresh[idx] = SquareItem()
        fresh[idx].state = m[idx].state
    return fresh


def _onepass_recording_entries(mem, monkeypatch, observe=None):
    """Pyramid.onepass with Base.nary_links wrapped by a recorder. Returns
    {(ci, cj): set of (candidate, cell)} - each base tile's simultaneous
    entries, over all round trips. observe: passed on to Pyramid.onepass."""
    by_realm = {}
    real = Base.nary_links
    real_init = Base.init_map

    def recorder(mem_, tile, entries):
        by_realm.setdefault(tile.realm, set()).update(entries)
        return real(mem_, tile, entries)

    def init_map(mem_):
        # onepass reruns the pass when the n-ary facts reveal a contradictory
        # candidate (2026-11-13) - only the last run's entries count
        by_realm.clear()
        return real_init(mem_)

    monkeypatch.setattr(Base, "nary_links", staticmethod(recorder))
    monkeypatch.setattr(Base, "init_map", staticmethod(init_map))
    Pyramid.onepass(mem, observe)
    monkeypatch.undo()
    base = mem.pyramid[0]
    return {idx: by_realm.get(tile.realm, set()) for idx, tile in np.ndenumerate(base)}


def _nary_in_isolation(mem, entries):
    """Base.nary_links for every base tile, with its own entries. Returns the
    tiles where it changed some cell."""
    return [idx for idx, tile in np.ndenumerate(mem.pyramid[0])
            if Base.nary_links(mem, tile, entries[idx])]


def _write(mem, entries, name, title, note):
    os.makedirs(HTML_DIR, exist_ok=True)
    return render_links_html(mem, os.path.join(HTML_DIR, name), title, entries, note)


def _write_candidates(mem, tile, name, title, note):
    """One page, one map per free candidate of base tile `tile` - what each
    candidate forces in the final map (2026-11-12)."""
    os.makedirs(HTML_DIR, exist_ok=True)
    return render_candidate_maps_html(mem, os.path.join(HTML_DIR, name), title, tile, note)


def _check_entries(mem, entries):
    """Every entry names a candidate, and its cell is a real cell id."""
    rows, cols = mem.map.shape
    for idx, tile_entries in entries.items():
        for candidate, cell in tile_entries:
            assert Pyramid.is_candidate(candidate), (idx, candidate)
            assert -1 <= cell[0] <= rows and -1 <= cell[1] <= cols, (idx, cell)


def test_nary_isolation_on_swept_maps(monkeypatch):
    """The 5-phase-swept 7x7-tile map (3x3 cells per tile, placement tiles
    (1+4I, 1+4J)) with several random seeds."""
    n, M = 7, 3
    for seed in SWEPT_SEEDS:
        m, N = build_swept_map(n, M, seed=seed)       # also sets the placement layout
        mem = DeviceMemory(_state_only(m), M)
        entries = _onepass_recording_entries(mem, monkeypatch)
        _check_entries(mem, entries)
        assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
        _write(mem, entries, f"nary_swept_seed{seed}.html", f"Swept map, seed {seed}",
               f"build_swept_map(n=7, M=3, seed={seed}), reduced to .state, then "
               "Base.init_map and one pass (Pyramid.onepass). Tiles list each base "
               "tile's links after the pass and its simultaneous entries over all "
               "round trips.")
        _write_candidates(mem, (5, 5), f"nary_swept_seed{seed}_tile55.html",
                          f"Swept map, seed {seed}: candidates of tile (5,5)",
                          f"build_swept_map(n=7, M=3, seed={seed}) after Base.init_map and one "
                          "pass. One map per free candidate of placement tile (5,5): the cells "
                          "it forces through single links, its chain, the triples starting at "
                          "it, and the conditions mentioning the candidate.")


def _frozen_area_map():
    """test_sudden_appearance.test_frozen_area's map after round 2: a 13x13
    board whose outermost ring is chosen (representation.build_margin_free_map),
    so the ring just inside it is blocked BY those chosen cells, around a
    free 9x9 core at (2,2)..(10,10); (5,5) and (6,5) placed by the old
    closure pipeline - the moment (9,8) is forced by the PAIR {(7,8), (7,9)}.

    Until 2026-11-14 the board was 11x11 with its border ring set blocked
    directly - no chosen cell behind it (Ivo: "there are blocked cells whose
    origin is unexplained. Change this"). Same states, one ring further
    out: every cell coordinate is the old one + 1, the tiles start at
    origin 2 instead of 1, tile indices are unchanged."""
    m = build_margin_free_map(9)
    do_closure(m, (5, 5), "round 1: (5,5) placed", display=None)
    do_closure(m, (6, 5), "round 2: (6,5) placed", display=None)
    return m


def test_nary_isolation_on_frozen_area(monkeypatch):
    """The frozen-area map: its free 9x9 core (origin 2; the chosen outer
    ring and the blocked ring it makes lie outside every tile) as 3x3 tiles
    of 3x3 cells, with tile (1,2) - rows 5-7, cols 8-10, where the old pipeline's rounds 3+ placed - as the
    placement tile (Ivo, 2026-11-10).

    Every free cell of (1,2) is then a candidate, so (7,8) and (7,9) are two
    candidates of the same placement tile. Their pair {(7,8), (7,9)} on
    (9,8) is kept (2026-11-13; until then Base.init_map pruned it): one cell
    per placement tile is the placement decision, and a seat the decision
    fills may lie in the same tile. On (9,8) there are also pairs with ONE
    member in (1,2) and one outside, e.g. {(7,7), (7,8)} and {(7,8), (9,7)}:
    the cross-tile n-ary situations. Candidate (7,8) forces (7,7), so in
    {(7,7), (7,8)} both terms are (7,8): the condition collapses into the new
    link (7,8) -> (9,8) (Base.nary_links). The ring around the core is
    blocked, so (9,8)
    forces the seat (10,6) (Base.blocked_corner_conditions).""" 
    Pyramid.set_placement_tiles({(1, 2)}, 3, 2)
    mem = DeviceMemory(_state_only(_frozen_area_map()), 3, origin=2)
    observe, steps = record_candidate_steps(mem, (5, 9))
    entries = _onepass_recording_entries(mem, monkeypatch, observe)

    pairs_on_98 = {frozenset(pos for pos, _s in c) for c in mem.map[9, 8].forced_by if len(c) == 2}
    assert frozenset({(7, 8), (7, 9)}) in pairs_on_98     # same-tile pair kept (2026-11-13)
    assert {frozenset({(7, 7), (7, 8)}), frozenset({(7, 8), (9, 7)})} <= pairs_on_98
    assert [lv.shape for lv in mem.pyramid] == [(3, 3), (1, 1)]
    _check_entries(mem, entries)
    assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
    # Ivo, 2026-11-10: candidate (7,8) forces (7,7), so (7,7) is an alias of
    # (7,8); (9,8)'s condition {(7,7), (7,8)} collapses to (7,8). Since
    # 2026-11-13 ("only T should get informed") this is no map link: it is
    # recorded in mem.collapsed and in the tile (2,2) that holds (9,8)
    singles_98 = {next(iter(c))[0] for c in mem.map[9, 8].forced_by if len(c) == 1}
    assert (7, 8) not in singles_98
    assert frozenset({(7, 7), (7, 8)}) in mem.collapsed[(7, 8), (9, 8)]
    # (2026-11-12, "there must be a seat at (10,6)"): (10,6)'s lower corners
    # are the blocked ring, so Base.init_map gives it single links from
    # (9,4), (9,6) and (9,8); the round trips repeat to convergence, and
    # (7,8)'s chain now runs (7,8) -> (9,8) -> (10,6) and (10,8) - the two
    # cells the old pipeline had in (9,8).forces1
    singles_106 = {next(iter(c))[0] for c in mem.map[10, 6].forced_by if len(c) == 1}
    assert {(9, 4), (9, 6), (9, 8)} <= singles_106
    assert ((7, 8), (9, 8), (10, 6)) in mem.pyramid[0][2, 2].links
    forced_by_78 = {(7, 8)}
    while True:
        more = {idx for idx in np.ndindex(mem.map.shape) if idx not in forced_by_78
                and any(len(c) == 1 and next(iter(c))[0] in forced_by_78 for c in mem.map[idx].forced_by)}
        more |= {y for (c, y) in mem.collapsed if c == (7, 8)} - forced_by_78
        if not more:
            break
        forced_by_78 |= more
    assert {(9, 8), (10, 6), (10, 8)} <= forced_by_78
    # partial collapse (2026-11-12, "partial collapses should be present in
    # .condition data"): a triple on (9,6) with two aliases of (7,8) and
    # (10,6) has shrunk to the pair {(7,8), (10,6)}, next to the original
    pairs_on_96 = {frozenset(pos for pos, _s in c) for c in mem.map[9, 6].forced_by if len(c) == 2}
    assert frozenset({(7, 8), (10, 6)}) in pairs_on_96
    _write(mem, entries, "nary_frozen_area.html", "Frozen area, after round 2",
           "test_sudden_appearance.test_frozen_area's board after round 2 ((5,5) and "
           "(6,5) placed), reduced to .state, then Base.init_map and one pass. Tiles "
           "start at origin 2; tile (1,2) is the placement tile. The pair {(7,8), (7,9)} "
           "on (9,8) names two candidates of one tile and is kept (one decision per tile, "
           "but a seat it fills may share the tile); pairs with one member in (1,2), like "
           "{(7,7), (7,8)}, are the cross-tile cases - switch on 'pair conditions'. "
           "(7,8) forces (7,7), so that pair collapses: (7,8) forces (9,8), recorded in "
           "tile (2,2); with the blocked ring, (9,8) forces the seat (10,6). (The outer ring "
           "is chosen; the blocked ring inside it comes from those chosen cells.)")
    _write_candidates(mem, (1, 2), "nary_frozen_area_tile12.html",
                      "Frozen area: candidates of tile (1,2)",
                      "The frozen-area board after round 2, after Base.init_map and one pass. "
                      "One map per free candidate of placement tile (1,2): the cells it forces "
                      "through single links, its chain, the triples starting at it, and the "
                      "conditions mentioning the candidate.")
    # every step of the pass for candidate (5,9) (Ivo, 2026-11-13)
    assert steps[-1]["forced"] == candidate_schema(mem, (5, 9))["forced"]   # the last step is the final state
    os.makedirs(HTML_DIR, exist_ok=True)
    render_candidate_steps_html(mem, os.path.join(HTML_DIR, "nary_frozen_area_steps_59.html"),
                                "Frozen area, placement tile (1,2): candidate (5,9), step by step", (5, 9), steps,
                                "One map per step of one pass (Pyramid.onepass): what (5,9) forces right "
                                "after that step. Cells newly forced in the step and new links of its chain "
                                "are highlighted.")



# --- suspended candidates (2026-11-12) -------------------------------------------------

def _singles(m):
    return {(next(iter(c))[0], idx) for idx in np.ndindex(m.shape) for c in m[idx].forced_by if len(c) == 1}


def test_contradictory_candidates_are_suspended():
    """Base.suspend_contradictory (Ivo, 2026-11-12: "If a cell A forces two
    cells that block each other, deprive A of its candidate status. This is
    a harmless postponement."): on the swept maps it suspends exactly the
    candidates whose own chain forces two cells that block each other -
    (3,5) with seed 0 (its chain reaches its own diagonal neighbours), (5,3)
    with seed 1, (16,15) and (16,17) with seed 2 (forced seats diagonal to
    each other). A suspended cell is no candidate for the pass, no base tile
    has a triple starting at it (not even as an outside cell linking in), and
    the n-ary step creates no link from it: seed 2's cascade from 32 / 202 links shrinks
    to a handful."""
    expected = {0: {(3, 5)}, 1: {(5, 3)}, 2: {(16, 15), (16, 17)}}
    for seed, suspended in expected.items():
        m, N = build_swept_map(7, 3, seed=seed)
        mem = DeviceMemory(_state_only(m), 3)
        Base.init_map(mem)
        before = _singles(mem.map)
        assert Base.suspend_contradictory(mem) == suspended, seed
        assert mem.suspended == suspended
        for a in suspended:
            assert Pyramid.in_placement_tile(a) and not Pyramid.is_candidate(a)
        Pyramid.upward(mem)
        Pyramid.downward(mem)
        new = _singles(mem.map) - before
        assert not {a for a, _y in new} & suspended, seed
        # a suspended cell leaves no trace: no base tile has a triple starting at it
        for tile in mem.pyramid[0].flat:
            assert not {t for t in tile.links if t[0] in suspended}, (seed, tile.realm)
    assert len(new) < 10    # seed 2, the last one


def test_suspension_lifts_once_a_conflicting_cell_is_chosen():
    """The suspension is not sticky (Ivo: "Once one of the blocked cells is
    chosen, A becomes eligible again"): it is derived from .state on every
    pass. Seed 2: (16,17) is suspended; once (15,17) - one of the seats its
    chain forces, diagonal to another one - is chosen, (16,17) is still free
    but no longer contradictory, and is a candidate again."""
    m, N = build_swept_map(7, 3, seed=2)
    fresh = _state_only(m)
    mem = DeviceMemory(fresh, 3)
    Base.init_map(mem)
    assert (16, 17) in Base.suspend_contradictory(mem)

    fresh[15, 17].state = StateEnum.chosen
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        y = (15 + dr, 17 + dc)
        if fresh[y].state == StateEnum.free:
            fresh[y].state = StateEnum.blocked
    mem = DeviceMemory(fresh, 3)
    Base.init_map(mem)
    assert fresh[16, 17].state == StateEnum.free
    assert (16, 17) not in Base.suspend_contradictory(mem)
    assert Pyramid.is_candidate((16, 17))


def test_nary_isolation_on_frozen_area_placement_tile_10(monkeypatch):
    """The frozen-area map with tile (1,0) - rows 5-7, cols 2-4, left of
    the chosen (5,5) and (6,5) - as the placement tile instead of (1,2)
    (Ivo, 2026-11-12: "display map with each candidate separately with
    placement tile (1,0)"). One pass (Base.init_map, suspension, up, down),
    then the map page and one map per candidate of (1,0)."""
    Pyramid.set_placement_tiles({(1, 0)}, 3, 2)
    mem = DeviceMemory(_state_only(_frozen_area_map()), 3, origin=2)
    observe, steps = record_candidate_steps(mem, (5, 2))
    entries = _onepass_recording_entries(mem, monkeypatch, observe)
    _check_entries(mem, entries)
    assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
    _write(mem, entries, "nary_frozen_area_p10.html", "Frozen area, placement tile (1,0)",
           "test_sudden_appearance.test_frozen_area's board after round 2, reduced to "
           ".state, with tile (1,0) (rows 5-7, cols 2-4) as the placement tile; then "
           "Base.init_map and one pass.")
    _write_candidates(mem, (1, 0), "nary_frozen_area_p10_tile10.html",
                      "Frozen area: candidates of placement tile (1,0)",
                      "The frozen-area board after round 2 with tile (1,0) as the placement "
                      "tile, after Base.init_map and one pass. One map per candidate: the cells "
                      "it forces through single links, its chain, the triples starting at it, "
                      "and the conditions mentioning the candidate.")
    # every step of the pass for candidate (5,2) (Ivo, 2026-11-13: "Create a
    # html with all steps, not just final state")
    assert steps[0]["label"] == "1. Base.init_map"
    assert steps[-1]["forced"] == candidate_schema(mem, (5, 2))["forced"]   # the last step is the final state
    os.makedirs(HTML_DIR, exist_ok=True)
    render_candidate_steps_html(mem, os.path.join(HTML_DIR, "nary_frozen_area_p10_steps_52.html"),
                                "Frozen area, placement tile (1,0): candidate (5,2), step by step", (5, 2), steps,
                                "One map per step of one pass (Pyramid.onepass): what (5,2) forces right "
                                "after that step. Cells newly forced in the step and new links of its chain "
                                "are highlighted.")


# --- soundness oracle (2026-11-13) ----------------------------------------------------

_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _real_block(shape, interior, p, q):
    """True if the 2x2 block with upper-left (p, q) lies on the board and
    needs covering: always without a margin (interior None), else only if
    its pixel (p+1, q+1) is in the real map (cores.Base.restrict_to_real_map)."""
    rows, cols = shape
    if not (0 <= p < rows - 1 and 0 <= q < cols - 1):
        return False
    if interior is None:
        return True
    (r0, r1), (c0, c1) = interior
    return r0 - 1 <= p <= r1 - 1 and c0 - 1 <= q <= c1 - 1


def _is_seat(st, shape, x, interior=None):
    """x is free (and eligible), and the three other corners of one of its
    real 2x2 blocks are blocked: x must be chosen."""
    if st[x] != StateEnum.free or (interior is not None and not Base.covers_real_map(interior, x)):
        return False
    for a, b in _DIAGONALS:
        corners = ((x[0] + a, x[1]), (x[0], x[1] + b), (x[0] + a, x[1] + b))
        if _real_block(shape, interior, min(x[0], x[0] + a), min(x[1], x[1] + b)) and \
                all(st[k] == StateEnum.blocked for k in corners):
            return True
    return False


def _simulate_forced(states, shape, c, return_states=False, interior=None):
    """What choosing c really forces on the board: mark c chosen (its diagonal
    neighbours blocked), then choose every seat - a free cell with the three
    other corners of one of its 2x2 blocks blocked - until none is left.
    return_states: also return the final states (forced, states).
    interior: the actual map's cells (2026-11-15) - then only eligible
    cells are seats, and only for blocks of the real map (_is_seat)."""
    st = dict(states)
    rows, cols = shape

    def choose(x):
        st[x] = StateEnum.chosen
        for a, b in _DIAGONALS:
            y = (x[0] + a, x[1] + b)
            if 0 <= y[0] < rows and 0 <= y[1] < cols and st[y] == StateEnum.free:
                st[y] = StateEnum.blocked

    def on_board_blocked(r, k):
        return 0 <= r < rows and 0 <= k < cols and st[(r, k)] == StateEnum.blocked

    choose(c)
    forced = set()
    while True:
        seats = [x for x in st if _is_seat(st, shape, x, interior)]
        if not seats:
            return (forced, st) if return_states else forced
        for x in seats:
            if st[x] == StateEnum.free:
                forced.add(x)
                choose(x)


def test_frozen_area_claims_match_simulation():
    """Soundness and completeness on the frozen area (2026-11-13, after full
    collapses started informing only tile T): for each free candidate c of
    the placement tile, what the pass says c forces - its single-link chain
    in the map plus its collapse targets (mem.collapsed), as
    links_html.candidate_schema shows it - equals what choosing c and
    filling seats to a fixed point forces on the real board."""
    for tile in ((1, 2), (1, 0)):
        Pyramid.set_placement_tiles({tile}, 3, 2)
        raw = _frozen_area_map()
        states = {idx: raw[idx].state for idx in np.ndindex(raw.shape)}
        mem = DeviceMemory(_state_only(raw), 3, origin=2)
        Pyramid.onepass(mem)
        r0, c0 = 2 + 3 * tile[0], 2 + 3 * tile[1]
        cands = [(r, k) for r in range(r0, r0 + 3) for k in range(c0, c0 + 3)
                 if Pyramid.is_candidate((r, k)) and mem.map[r, k].state == StateEnum.free]
        assert cands, tile
        for c in cands:
            claimed = {tuple(int(v) for v in k.split(",")) for k in candidate_schema(mem, c)["forced"]}
            assert claimed == _simulate_forced(states, raw.shape, c), (tile, c)


# --- the frozen area with a soft ("zipper") margin (2026-11-15) ------------------------

# the actual map's cells on the 13x13 zipper board: the 9x9 core (ends exclusive)
ZIPPER_INTERIOR = ((2, 11), (2, 11))
# the tile lattice over the whole board (Ivo: "extend the lattice of tiles to
# contain the margin cells too. however keep the placement tiles at their old
# position"): origin -1, the edge tiles cut to the board (DeviceMemory clip);
# a tile that was (ci, cj) at origin 2 is (ci+1, cj+1) now, same cells
ZIPPER_ORIGIN = -1


def _zipper_area_map(extra=()):
    """The frozen-area scenario with a soft margin (Ivo, 2026-11-15: "A margin
    as a solid wall of chosen cells has an annoying property: Just a few more
    placements can 'freeze' large areas, that is, cells can no longer be
    chosen independently. ... In the line defining the margin, leave out
    every second placement"): representation.build_zipper_margin_map(9) -
    13x13, every second cell of the outer ring chosen, every second cell of
    the ring inside it blocked by them - then (5,5) and (6,5) placed by the
    old closure pipeline, as in _frozen_area_map. No seat to fill.

    The actual map is the 9x9 core (ZIPPER_INTERIOR); the free cells of the
    ring around it that share an edge with the core are half-cells - their
    square has a 1x2 / 2x1 inside the actual map - and may be chosen too.
    extra: further cells to choose, each followed by filling the seats it
    makes (on the actual map)."""
    m = build_zipper_margin_map(9)
    do_closure(m, (5, 5), "round 1: (5,5) placed", display=None)
    do_closure(m, (6, 5), "round 2: (6,5) placed", display=None)
    m = _state_only(m)
    for x in extra:
        states = {idx: m[idx].state for idx in np.ndindex(m.shape)}
        _forced, final = _simulate_forced(states, m.shape, x, return_states=True, interior=ZIPPER_INTERIOR)
        for idx, st in final.items():
            m[idx].state = st
    return m


def _zipper_pass(raw, tile, monkeypatch):
    """One pass on a zipper board with placement tile `tile` (lattice at
    ZIPPER_ORIGIN, clipped, ZIPPER_INTERIOR), checked: fixed point of the
    round trips, every candidate of the tile against the simulation.
    Returns (mem, entries, {candidate: forced cells})."""
    Pyramid.set_placement_tiles({tile}, 3, ZIPPER_ORIGIN)
    mem = DeviceMemory(_state_only(raw), 3, origin=ZIPPER_ORIGIN, clip=True, interior=ZIPPER_INTERIOR)
    entries = _onepass_recording_entries(mem, monkeypatch)
    _check_entries(mem, entries)
    assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
    assert [lv.shape for lv in mem.pyramid] == [(5, 5), (2, 2), (1, 1)]
    states = {idx: raw[idx].state for idx in np.ndindex(raw.shape)}
    r0, c0 = ZIPPER_ORIGIN + 3 * tile[0], ZIPPER_ORIGIN + 3 * tile[1]
    cands = [(r, k) for r in range(r0, r0 + 3) for k in range(c0, c0 + 3)
             if Pyramid.is_candidate((r, k)) and mem.map[r, k].state == StateEnum.free]
    assert cands
    forced = {}
    for c in cands:
        forced[c] = _simulate_forced(states, raw.shape, c, interior=ZIPPER_INTERIOR)
        claimed = {tuple(int(v) for v in k.split(",")) for k in candidate_schema(mem, c)["forced"]}
        assert claimed == forced[c], c
    return mem, entries, forced


def _is_half_cell(x):
    (r0, r1), (c0, c1) = ZIPPER_INTERIOR
    return Base.eligible(ZIPPER_INTERIOR, x) and not (r0 <= x[0] < r1 and c0 <= x[1] < c1)


def test_nary_isolation_on_zipper_area(monkeypatch):
    """The two frozen-area placement layouts - old tiles (1,0) and (1,2),
    now (2,1) and (2,3) on the lattice that covers the margin too - on the
    zipper-margin board: one pass each, the map page and the candidate
    pages (the "for correctness" display); every candidate's claims equal
    the simulation."""
    raw = _zipper_area_map()
    for tile, pages in (((2, 1), "p10"), ((2, 3), "p12")):
        mem, entries, _forced = _zipper_pass(raw, tile, monkeypatch)
        r0, c0 = ZIPPER_ORIGIN + 3 * tile[0], ZIPPER_ORIGIN + 3 * tile[1]
        rows = f"rows {r0}-{r0 + 2}, cols {c0}-{c0 + 2}"
        _write(mem, entries, f"nary_zipper_area_{pages}.html",
               f"Zipper-margin area, placement tile ({tile[0]},{tile[1]})",
               "The frozen-area scenario with a soft (zipper) margin instead of a solid wall: of "
               "the outer ring only every second cell is chosen, so only every second cell of the "
               "ring inside it is blocked; then (5,5) and (6,5) placed. Reduced to .state. The tile "
               "lattice covers the margin too (origin -1, edge tiles cut to the board); the actual "
               "map is the 9x9 core, the free cells bordering it are half-cells and may be chosen. "
               f"Placement tile ({tile[0]},{tile[1]}) ({rows}); Base.init_map and one pass.")
        _write_candidates(mem, tile, f"nary_zipper_area_{pages}_tile{tile[0]}{tile[1]}.html",
                          f"Zipper-margin area: candidates of placement tile ({tile[0]},{tile[1]})",
                          "The frozen-area scenario with a zipper margin, after Base.init_map and one "
                          "pass. One map per candidate: the cells it forces, its chain, the triples "
                          "starting at it, and the conditions mentioning the candidate.")


def test_half_cells_can_be_forced(monkeypatch):
    """Half-cells forced (Ivo, 2026-11-15: "We now allow the selection of
    half-cells at the margin. Recompute the zipper maps and allow half-cells
    to be forced"): on the zipper board, the plain scenario makes no
    candidate force one; one more placement, (8,4), does - candidate (5,3)
    of tile (2,1) forces the half-cells (4,1) and (6,1), candidate (7,8) of
    tile (2,3) the half-cells (4,11), (6,11), (8,11), (10,11), neither
    contradictory. The pass finds them, as the simulation does. (With
    (2,5) instead, (5,2) would force the half-cell (2,1) too - but it also
    leaves the real block at (3,1) uncoverable, and the pass rightly
    suspends it after the n-ary step.)"""
    raw = _zipper_area_map(extra=[(8, 4)])
    for tile, pages, cand, halves in (((2, 1), "p10", (5, 3), {(4, 1), (6, 1)}),
                                      ((2, 3), "p12", (7, 8), {(4, 11), (6, 11), (8, 11), (10, 11)})):
        mem, entries, forced = _zipper_pass(raw, tile, monkeypatch)
        assert {x for x in forced[cand] if _is_half_cell(x)} == halves
        _write(mem, entries, f"nary_zipper_halfcells_{pages}.html",
               f"Zipper-margin area with (8,4) placed, placement tile ({tile[0]},{tile[1]})",
               "The zipper-margin frozen area with one more placement, (8,4), and its seats filled. "
               f"Candidate ({cand[0]},{cand[1]}) now forces the half-cell(s) "
               f"{', '.join(map(str, sorted(halves)))} - cells of the ring around the actual map whose "
               "square has a 1x2 / 2x1 inside it.")
        _write_candidates(mem, tile, f"nary_zipper_halfcells_{pages}_tile{tile[0]}{tile[1]}.html",
                          f"Zipper-margin area with (8,4) placed: candidates of tile ({tile[0]},{tile[1]})",
                          f"One map per candidate; ({cand[0]},{cand[1]}) forces half-cells outside the "
                          "actual map.")
