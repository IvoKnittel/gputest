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
from links_html import render_links_html, render_candidate_maps_html
from map_of_squares import SquareItem, StateEnum
from representation import build_map_of_squares, place_blocked_squares
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


def _onepass_recording_entries(mem, monkeypatch):
    """Pyramid.onepass with Base.nary_links wrapped by a recorder. Returns
    {(ci, cj): set of (candidate, cell)} - each base tile's simultaneous
    entries, over all round trips."""
    by_realm = {}
    real = Base.nary_links

    def recorder(mem_, tile, entries):
        by_realm.setdefault(tile.realm, set()).update(entries)
        return real(mem_, tile, entries)

    monkeypatch.setattr(Base, "nary_links", staticmethod(recorder))
    Pyramid.onepass(mem)
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
                          "it, and the conditions naming it.")


def _frozen_area_map():
    """test_sudden_appearance.test_frozen_area's map after round 2: an 11x11
    board, border blocked, (4,4) and (5,4) placed by the old closure
    pipeline - the moment (8,7) is forced by the PAIR {(6,7), (6,8)}."""
    m = build_map_of_squares(11, 11)
    border = [(i, j) for i in range(11) for j in range(11) if i in (0, 10) or j in (0, 10)]
    place_blocked_squares(m, border)
    do_closure(m, (4, 4), "round 1: (4,4) placed", display=None)
    do_closure(m, (5, 4), "round 2: (5,4) placed", display=None)
    return m


def test_nary_isolation_on_frozen_area(monkeypatch):
    """The frozen-area map: its inner 9x9 cells (origin 1, the blocked border
    outside every tile) as 3x3 tiles of 3x3 cells, with tile (1,2) - rows
    4-6, cols 7-9, where the old pipeline's rounds 3+ placed - as the
    placement tile (Ivo, 2026-11-10).

    Every free cell of (1,2) is then a candidate, so (6,7) and (6,8) are two
    candidates of the same placement tile, and Base.init_map prunes the old
    pipeline's pair {(6,7), (6,8)} on (8,7) - one placement tile only ever
    gets one chosen cell. What stays on (8,7) are pairs with ONE member in
    (1,2) and one outside, e.g. {(6,6), (6,7)} and {(6,7), (8,6)}: the
    cross-tile n-ary situations. Candidate (6,7) forces (6,6), so in
    {(6,6), (6,7)} both terms are (6,7): the condition collapses into the new
    link (6,7) -> (8,7) (Base.nary_links). The border is blocked, so (8,7)
    forces the seat (9,5) (Base.blocked_corner_conditions).""" 
    Pyramid.set_placement_tiles({(1, 2)}, 3, 1)
    mem = DeviceMemory(_state_only(_frozen_area_map()), 3, origin=1)
    entries = _onepass_recording_entries(mem, monkeypatch)

    pairs_on_87 = {frozenset(pos for pos, _s in c) for c in mem.map[8, 7].forced_by if len(c) == 2}
    assert frozenset({(6, 7), (6, 8)}) not in pairs_on_87
    assert {frozenset({(6, 6), (6, 7)}), frozenset({(6, 7), (8, 6)})} <= pairs_on_87
    assert [lv.shape for lv in mem.pyramid] == [(3, 3), (1, 1)]
    _check_entries(mem, entries)
    assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
    # Ivo, 2026-11-10: candidate (6,7) forces (6,6), so (6,6) is an alias of
    # (6,7); (8,7)'s condition {(6,6), (6,7)} collapses to (6,7) - a new link
    singles_87 = {next(iter(c))[0] for c in mem.map[8, 7].forced_by if len(c) == 1}
    assert (6, 7) in singles_87
    # (2026-11-12, "there must be a seat at (9,5)"): (9,5)'s lower corners
    # are the blocked border, so Base.init_map gives it single links from
    # (8,3), (8,5) and (8,7); the round trips repeat to convergence, and
    # (6,7)'s chain now runs (6,7) -> (8,7) -> (9,5) and (9,7) - the two
    # cells the old pipeline had in (8,7).forces1
    singles_95 = {next(iter(c))[0] for c in mem.map[9, 5].forced_by if len(c) == 1}
    assert {(8, 3), (8, 5), (8, 7)} <= singles_95
    assert ((6, 7), (8, 7), (9, 5)) in mem.pyramid[0][2, 2].links
    forced_by_67 = {(6, 7)}
    while True:
        more = {idx for idx in np.ndindex(mem.map.shape) if idx not in forced_by_67
                and any(len(c) == 1 and next(iter(c))[0] in forced_by_67 for c in mem.map[idx].forced_by)}
        if not more:
            break
        forced_by_67 |= more
    assert {(8, 7), (9, 5), (9, 7)} <= forced_by_67
    # partial collapse (2026-11-12, "partial collapses should be present in
    # .condition data"): a triple on (8,5) with two aliases of (6,7) and
    # (9,5) has shrunk to the pair {(6,7), (9,5)}, next to the original
    pairs_on_85 = {frozenset(pos for pos, _s in c) for c in mem.map[8, 5].forced_by if len(c) == 2}
    assert frozenset({(6, 7), (9, 5)}) in pairs_on_85
    _write(mem, entries, "nary_frozen_area.html", "Frozen area, after round 2",
           "test_sudden_appearance.test_frozen_area's board after round 2 ((4,4) and "
           "(5,4) placed), reduced to .state, then Base.init_map and one pass. Tiles "
           "start at origin 1; tile (1,2) is the placement tile, so the old pair "
           "{(6,7), (6,8)} on (8,7) is pruned (two candidates of one tile); pairs with "
           "one member in (1,2), like {(6,6), (6,7)}, remain - switch on 'pair conditions'. "
           "(6,7) forces (6,6), so that pair collapses into the new link (6,7) -> (8,7); "
           "with the blocked border, (8,7) forces the seat (9,5).")
    _write_candidates(mem, (1, 2), "nary_frozen_area_tile12.html",
                      "Frozen area: candidates of tile (1,2)",
                      "The frozen-area board after round 2, after Base.init_map and one pass. "
                      "One map per free candidate of placement tile (1,2): the cells it forces "
                      "through single links, its chain, the triples starting at it, and the "
                      "conditions naming it.")



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
    """The frozen-area map with tile (1,0) - rows 4-6, cols 1-3, left of
    the chosen (4,4) and (5,4) - as the placement tile instead of (1,2)
    (Ivo, 2026-11-12: "display map with each candidate separately with
    placement tile (1,0)"). One pass (Base.init_map, suspension, up, down),
    then the map page and one map per candidate of (1,0)."""
    Pyramid.set_placement_tiles({(1, 0)}, 3, 1)
    mem = DeviceMemory(_state_only(_frozen_area_map()), 3, origin=1)
    entries = _onepass_recording_entries(mem, monkeypatch)
    _check_entries(mem, entries)
    assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
    _write(mem, entries, "nary_frozen_area_p10.html", "Frozen area, placement tile (1,0)",
           "test_sudden_appearance.test_frozen_area's board after round 2, reduced to "
           ".state, with tile (1,0) (rows 4-6, cols 1-3) as the placement tile; then "
           "Base.init_map and one pass.")
    _write_candidates(mem, (1, 0), "nary_frozen_area_p10_tile10.html",
                      "Frozen area: candidates of placement tile (1,0)",
                      "The frozen-area board after round 2 with tile (1,0) as the placement "
                      "tile, after Base.init_map and one pass. One map per candidate: the cells "
                      "it forces through single links, its chain, the triples starting at it, "
                      "and the conditions naming it.")
