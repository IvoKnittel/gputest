"""The maps with a soft ("zipper") margin (2026-11-15, Ivo: "now do the
zipper margin version of Swept map, and the supertile-not-supersupertile
versions of frozen area and swept maps").

Zipper margin: of the line defining the margin only every second cell is
chosen (cells with row + col even), so only every second cell next to it is
blocked - a solid wall of chosen cells "freezes" large areas after a few
more placements (Ivo, same day). See representation.build_zipper_margin_map
and test_context_tiles.build_framed_map(zipper=True).

Pages, in html_out/zipper/, each a map page plus one map per candidate (the
"for correctness" display); the superlattice ones also the entries page:
- swept maps (seeds 0/1/2), build_swept_map(zipper=True), open seats filled
  first (the sweep places without checking), uncompletable boards discarded:
  - supersuperlattice: placement tiles (1+4I, 1+4J) - swept_seed{s}_supersuper;
  - superlattice: placement tiles (1+2I, 1+2J) - swept_seed{s}_superlattice;
- the frozen area on the zipper board (test_nary_isolation._zipper_area_map),
  superlattice tiles (0,0), (0,2), (2,0), (2,2) - frozen_area_superlattice.
Every pass is checked as in test_nary_superlattice: fixed point of the
round trips, every candidate's claims equal to the board simulation.

Half-cells (2026-11-15, Ivo: "We now allow the selection of half-cells at
the margin"): the actual map is the interior; the cells bordering it are
half-cells - eligible, they may be forced (cores.Base.eligible /
restrict_to_real_map, DeviceMemory interior) - and only blocks of the real
map need covering. Swept maps: interior rows/cols 1..N-2, the frame row is
the half-cell row. Frozen area: the tile lattice covers the margin too
(origin -1, edge tiles cut to the board), placement tiles on the same cells
as before - (0,0), (0,2), (2,0), (2,2) at origin 2 are (1,1), (1,3), (3,1),
(3,3) now.
"""
import os

import numpy as np

from links_html import render_links_html, render_candidate_maps_html, render_entries_html
from test_context_tiles import build_swept_map, placement_core_indices
from test_nary_isolation import (HTML_DIR, SWEPT_SEEDS, _state_only, _zipper_area_map,
                                 ZIPPER_INTERIOR, ZIPPER_ORIGIN)
from test_nary_superlattice import completable_swept_map, superlattice_tiles, _run, _fill_open_seats, _holes
from map_of_squares import StateEnum
from cores import Base
from representation import build_zipper_margin_map

ZIPPER_DIR = os.path.join(HTML_DIR, "zipper")


def _pages(mem, entries, tiles, name, title, note, with_entries):
    os.makedirs(ZIPPER_DIR, exist_ok=True)
    render_links_html(mem, os.path.join(ZIPPER_DIR, name + ".html"), title, entries, note)
    render_candidate_maps_html(mem, os.path.join(ZIPPER_DIR, name + "_candidates.html"),
                               title + ": every candidate", sorted(tiles),
                               note + " One map per free candidate of every placement tile - what it "
                               "forces on its own; its tile is outlined bold.")
    if with_entries:
        render_entries_html(mem, os.path.join(ZIPPER_DIR, name + "_entries.html"), title + ": entries per tile",
                            note + " Dots: where each candidate's chain enters a tile, one colour per "
                            "placement tile, one shade per candidate.")


def swept_interior(N):
    """The actual map of an N x N swept board: rows/cols 1..N-2 (ends
    exclusive below); the frame row/column is the half-cell row."""
    return ((1, N - 1), (1, N - 1))


def zipper_swept_map(seed):
    """build_swept_map(7, 3, seed, zipper=True), reduced to .state, its open
    seats filled (on the actual map); None if that leaves a real 2x2 block
    nothing can cover. Returns (raw, filled, interior) or None."""
    m, N = build_swept_map(7, 3, seed=seed, zipper=True)
    raw = _state_only(m)
    interior = swept_interior(N)
    filled = _fill_open_seats(raw, interior=interior)
    if _holes({idx: raw[idx].state for idx in np.ndindex(raw.shape)}, raw.shape, interior):
        return None
    return raw, filled, interior


def test_zipper_swept_maps(monkeypatch):
    """The swept maps with a zipper frame, on the supersuperlattice and on
    the superlattice. None is discarded (with the solid frame, seed 0 was)."""
    for seed in SWEPT_SEEDS:
        board = zipper_swept_map(seed)
        assert board is not None, seed
        raw, filled, interior = board
        for lattice, tiles in (("supersuper", placement_core_indices(7)),
                               ("superlattice", superlattice_tiles(7, 7, (1, 1)))):
            mem, entries, contradictory = _run(raw, tiles, 3, 0, monkeypatch, interior=interior)
            assert contradictory == []
            layout = ("(1+4I, 1+4J) - the supersuperlattice, 4 tiles" if lattice == "supersuper"
                      else "(1+2I, 1+2J) - the superlattice, 9 tiles")
            _pages(mem, entries, tiles, f"swept_seed{seed}_{lattice}",
                   f"Zipper swept map seed {seed}, {lattice} placement tiles",
                   f"build_swept_map(n=7, M=3, seed={seed}, zipper=True): the frame chooses only every "
                   f"second cell; the frame row is the half-cell row around the actual map. Reduced "
                   f"to .state; its {len(filled)} open seats filled first; placement "
                   f"tiles {layout}, all candidates in one pass; Base.init_map and one pass."
                   + (f" Suspended: {', '.join(map(str, sorted(mem.suspended)))}." if mem.suspended else ""),
                   with_entries=lattice == "superlattice")


def test_zipper_frozen_area_superlattice(monkeypatch):
    """The frozen-area scenario on the zipper board with the superlattice
    tiles as placement tiles - (1,1), (1,3), (3,1), (3,3) on the lattice
    that covers the margin (origin -1, clipped), the cells (0,0), (0,2),
    (2,0), (2,2) were at origin 2."""
    raw = _state_only(_zipper_area_map())
    tiles = superlattice_tiles(5, 5, (1, 1))
    assert tiles == {(1, 1), (1, 3), (3, 1), (3, 3)}
    mem, entries, contradictory = _run(raw, tiles, 3, ZIPPER_ORIGIN, monkeypatch, clip=True,
                                       interior=ZIPPER_INTERIOR)
    assert contradictory == [] and mem.suspended == set()
    _pages(mem, entries, tiles, "frozen_area_superlattice",
           "Zipper frozen area, superlattice placement tiles",
           "The frozen-area scenario ((5,5) and (6,5) placed) with a zipper margin: of the outer ring "
           "only every second cell is chosen. Reduced to .state. The tile lattice covers the margin "
           "too (origin -1, edge tiles cut to the board); the actual map is the 9x9 core, the free "
           "cells bordering it are half-cells and may be chosen. Placement tiles (1,1), (1,3), (3,1), "
           "(3,3) - four tiles, all candidates in one pass; Base.init_map and one pass.",
           with_entries=True)


def test_quarter_cells_can_be_forced(monkeypatch):
    """A free quarter-cell at a corner of the actual map (Ivo, 2026-11-15, on
    such a cell: "I see no problem here. It becomes an id_in of the margin
    tile and may also force an exit on the map"): the zipper margin with
    the other phase (build_zipper_margin_map(9, phase=1) - ring cells with
    i + j odd chosen) leaves the ring-1 corners (1,1), (1,11), (11,1),
    (11,11) free; each square has 1x1 inside the actual map. The frozen
    scenario's (5,5) and (6,5) placed (no seat to fill), superlattice
    placement tiles: candidates (3,3), (3,9), (9,3), (9,9) each force the
    quarter-cell next to them - e.g. (3,3) blocks (2,2), the last cell of
    the block at (1,1) besides (1,1) itself. Pass = simulation."""
    m = _state_only(build_zipper_margin_map(9, phase=1))
    for p in [(5, 5), (6, 5)]:
        m[p].state = StateEnum.chosen
        for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            y = (p[0] + a, p[1] + b)
            if m[y].state == StateEnum.free:
                m[y].state = StateEnum.blocked
    assert _fill_open_seats(m, interior=ZIPPER_INTERIOR) == []
    quarter = {(1, 1), (1, 11), (11, 1), (11, 11)}
    assert all(m[q].state == StateEnum.free for q in quarter)
    tiles = superlattice_tiles(5, 5, (1, 1))
    mem, entries, contradictory = _run(m, tiles, 3, ZIPPER_ORIGIN, monkeypatch, clip=True,
                                       interior=ZIPPER_INTERIOR)
    assert contradictory == []
    for cand, q in (((3, 3), (1, 1)), ((3, 9), (1, 11)), ((9, 3), (11, 1)), ((9, 9), (11, 11))):
        assert q in Base.claimed(mem, cand), cand
    _pages(mem, entries, tiles, "frozen_area_phase1_superlattice",
           "Zipper frozen area (other phase), superlattice placement tiles",
           "The frozen-area scenario with the zipper margin's other phase: ring cells with i + j odd "
           "chosen, so the corners of the ring around the actual map - (1,1), (1,11), (11,1), (11,11) - "
           "stay free; their squares have a 1x1 quarter inside the actual map. Candidates (3,3), (3,9), "
           "(9,3), (9,9) each force the quarter-cell next to them. Tile lattice over the margin (origin "
           "-1, clipped); placement tiles (1,1), (1,3), (3,1), (3,3).",
           with_entries=True)
