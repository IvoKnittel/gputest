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
from links_html import render_links_html
from map_of_squares import SquareItem
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
    link (6,7) -> (8,7) (Base.nary_links).""" 
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
    assert ((6, 7), DUMMY, DUMMY) in mem.pyramid[0][2, 2].links
    _write(mem, entries, "nary_frozen_area.html", "Frozen area, after round 2",
           "test_sudden_appearance.test_frozen_area's board after round 2 ((4,4) and "
           "(5,4) placed), reduced to .state, then Base.init_map and one pass. Tiles "
           "start at origin 1; tile (1,2) is the placement tile, so the old pair "
           "{(6,7), (6,8)} on (8,7) is pruned (two candidates of one tile); pairs with "
           "one member in (1,2), like {(6,6), (6,7)}, remain - switch on 'pair conditions'. "
           "(6,7) forces (6,6), so that pair collapses into the new link (6,7) -> (8,7).")
