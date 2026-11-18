"""Blocking and self-blocking, marked on the maps with full-closed margins
(2026-11-16, Ivo: "For now, we return to the full-closed margins of swept
map and frozen area map. Even though a full-closed wall is not a good choice
for a margin, it can arise naturally and we have to deal with it. And it is
just right for our testing. In the full-closed-margin swept map and frozen
area map (final maps without the not-yet-implemented n-ary links), find a
situation where a path from a candidate blocks the path of a candidate from
another placement tile, and a candidate that is self-blocking. Mark the
blocking cells and the blocked candidates in the maps.")

"Final maps": after one pass (Pyramid.onepass) - the n-ary facts it has,
no compounds. Situations found (a survey of the layouts):
- swept map seed 1 (build_swept_map, open seats filled - seed 0 has no
  valid completion), superlattice placement tiles: 4 blocking pairs and 3
  self-blocking candidates, (9,11), (9,15), (11,15);
- frozen area (build_margin_free_map), placement tiles (1,0) and (1,2) -
  its two layouts so far, together: 10 blocking pairs. No placement layout
  of the frozen area has a self-blocking candidate.
Pages in html_out/blocking/ (links_html.render_blocking_html).
"""
import os

import numpy as np

from cores import Base
from map_of_squares import StateEnum
from links_html import render_blocking_html
from test_nary_isolation import HTML_DIR, _state_only, _frozen_area_map
from test_nary_superlattice import completable_swept_map, superlattice_tiles, _run, _conflict_when_both_chosen

BLOCKING_DIR = os.path.join(HTML_DIR, "blocking")


def _situations(mem):
    """Blocking pairs from the tiles' .blocked lists; self-blocking
    candidates - the pass's report mem.refuted, suspended in the pass and
    entered as (c, c)."""
    blocked = set().union(*(t.blocked for t in mem.pyramid[0].flat))
    assert {p[0] for p in blocked if p[0] == p[1]} == set(mem.refuted)
    return sorted(p for p in blocked if p[0] != p[1]), sorted(mem.refuted)


def _check(mem, raw):
    """Every blocking pair has blocking cells and is a real conflict on the
    board; every refuted candidate is suspended, still free (the pass is
    analysis only), and has its blocking cells recorded, all of them in its
    chain."""
    pairs, selfs = _situations(mem)
    states = {idx: raw[idx].state for idx in np.ndindex(raw.shape)}
    for c1, c2 in pairs:
        assert Base.blocking_cells(mem, c1, c2), (c1, c2)
        assert _conflict_when_both_chosen(states, raw.shape, (c1, c2)), (c1, c2)
    for c in selfs:
        assert c in mem.suspended and mem.map[c].state == StateEnum.free
        chain, cell_pairs = mem.refuted[c]
        assert cell_pairs and all(x in chain and y in chain for x, y in cell_pairs), c
    return pairs, selfs


def test_blocking_swept_map(monkeypatch):
    raw, filled = completable_swept_map(7, 3, 1)
    tiles = superlattice_tiles(7, 7, (1, 1))
    mem, _entries, _c = _run(raw, tiles, 3, 0, monkeypatch)
    pairs, selfs = _check(mem, raw)
    assert pairs == [((3, 5), (5, 9)), ((3, 15), (5, 11)), ((11, 9), (15, 11)), ((11, 17), (15, 15))]
    assert selfs == [(9, 11), (9, 15), (11, 15)]
    # each chain stops at its first contradiction (Base.chain_until_contradiction):
    # (9,15) forces the seats (8,15) and (7,16), which block each other - nothing
    # derived from placing both counts; (11,15) runs into the same pair; the
    # n-ary step stops collapsing for (9,11) at its first contradiction
    assert mem.refuted[(9, 15)] == (frozenset({(9, 15), (8, 15), (7, 16)}), frozenset({((7, 16), (8, 15))}))
    assert mem.refuted[(11, 15)][1] == frozenset({((7, 16), (8, 15))})
    assert mem.refuted[(9, 11)][1] == frozenset({((7, 16), (8, 15)), ((8, 16), (9, 15))})
    assert len(mem.refuted[(9, 11)][0]) == 8
    os.makedirs(BLOCKING_DIR, exist_ok=True)
    render_blocking_html(mem, os.path.join(BLOCKING_DIR, "swept_seed1.html"),
                         "Swept map seed 1: blocking and self-blocking",
                         f"build_swept_map(n=7, M=3, seed=1) with its solid frame, reduced to .state, its "
                         f"{len(filled)} open seats filled; superlattice placement tiles (1+2I, 1+2J); one pass "
                         "(no compounds, analysis only). One map per blocking pair of candidates from different "
                         "placement tiles, and per refuted candidate - suspended in the pass, reported for the "
                         "CPU's unblocking step. (9,11) is refuted only through its n-ary facts. Each chain is "
                         "shown up to its first contradiction only.")


def test_unblocking_step_swept_map(monkeypatch):
    """The clean split (2026-11-16): the pass reports, the CPU's placement
    step blocks (Pyramid.unblock) and fills the seats that makes, then the
    next pass. Seed 1: one round blocks all three refuted candidates and
    fills (10,12), (10,14); the next pass refutes nothing."""
    from cores import Pyramid, DeviceMemory
    from test_nary_isolation import _state_only as state_only
    raw, _filled = completable_swept_map(7, 3, 1)
    Pyramid.set_placement_tiles(superlattice_tiles(7, 7, (1, 1)), 3, 0)
    mem = DeviceMemory(state_only(raw), 3)
    assert Pyramid.analyse_and_unblock(mem) == 1
    (before, refuted, seats), = mem.unblock_rounds
    assert sorted(refuted) == [(9, 11), (9, 15), (11, 15)] and seats == [(10, 12), (10, 14)]
    assert all(before[c] == StateEnum.free and mem.map[c].state == StateEnum.blocked for c in refuted)
    assert mem.refuted == {} and mem.suspended == set()


def test_blocking_frozen_area(monkeypatch):
    raw = _state_only(_frozen_area_map())
    mem, _entries, _c = _run(raw, {(1, 0), (1, 2)}, 3, 2, monkeypatch)
    pairs, selfs = _check(mem, raw)
    assert len(pairs) == 10 and selfs == []
    os.makedirs(BLOCKING_DIR, exist_ok=True)
    render_blocking_html(mem, os.path.join(BLOCKING_DIR, "frozen_area.html"),
                         "Frozen area: blocking",
                         "The frozen area with its solid margin (13x13, outer ring chosen), (5,5) and (6,5) "
                         "placed; placement tiles (1,0) and (1,2) together (origin 2); one pass (no "
                         "compounds). One map per blocking pair of candidates from the two placement tiles. "
                         "No placement layout of the frozen area has a self-blocking candidate.")
