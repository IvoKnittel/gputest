"""N-ary link generation with placement tiles on the SUPERLATTICE (2026-11-13,
Ivo: "So far, we have used the supersuper lattice for placement tiles. Now
create a new subfolder and create new htmls using the superlattice. Then we
have several candidates placed simultaneously").

Supersuperlattice (test_context_tiles.placement_core_indices, all other
tests so far): placement tiles at (1+4I, 1+4J) - period 4 tiles, as the 16
colour codes of Quality/image_to_squares.tile_upper_left_indices(super=True).
Superlattice (here): period 2 tiles - one shift phase of the 4-colour sweep,
e.g. (1+2I, 1+2J). Placement tiles are then only one transit tile apart,
so candidates of different placement tiles - placed simultaneously in one
pass - meet in each other's conditions far more often.

Maps:
- the swept maps (build_swept_map, seeds 0/1/2), odd-odd phase: 9 placement
  tiles. The sweep that builds them places without any check and never
  fills the seats its placements create, so the open seats are filled first
  (_fill_open_seats) - they are forced by the current state alone, before
  any candidate is chosen. A map left with a 2x2 block nothing can cover is
  discarded (Ivo, 2026-11-13: "the original placement without check is
  wrong. for now, just check for this and discard") - seed 0;
- the frozen area (origin 2, 3x3 tiles), even-even phase: tiles (0,0),
  (0,2), (2,0), (2,2) - the only phase with more than one tile there.

Each test runs one pass, checks the n-ary step reached its fixed point,
compares every candidate's claims with the board simulation, and writes
the map page and one page with a map per candidate of every placement tile
into html_out/superlattice/.
"""
import os

import numpy as np

from cores import DeviceMemory, Pyramid, Base
from links_html import (render_links_html, render_candidate_maps_html, candidate_schema, _contradiction,
                        render_open_seats_html, render_entries_html, candidate_entries)
from map_of_squares import StateEnum
from test_context_tiles import build_swept_map
from test_nary_isolation import (HTML_DIR, SWEPT_SEEDS, _DIAGONALS, _real_block, _is_seat, _state_only, _onepass_recording_entries,
                                 _check_entries, _nary_in_isolation, _frozen_area_map, _simulate_forced)

SUPER_DIR = os.path.join(HTML_DIR, "superlattice")


def superlattice_tiles(n_rows, n_cols, phase):
    """Tile indices of one superlattice shift phase: (phase[0] + 2I,
    phase[1] + 2J) within an n_rows x n_cols tile grid."""
    return {(ci, cj) for ci in range(phase[0], n_rows, 2) for cj in range(phase[1], n_cols, 2)}


def _seat_block(m, x, interior=None):
    """The three other corners of a (real, see _real_block) 2x2 block of x
    that are all blocked in the state-only map m (x is a seat), or None."""
    if interior is not None and not Base.covers_real_map(interior, x):
        return None
    for a, b in _DIAGONALS:
        corners = ((x[0] + a, x[1]), (x[0], x[1] + b), (x[0] + a, x[1] + b))
        if _real_block(m.shape, interior, min(x[0], x[0] + a), min(x[1], x[1] + b)) and \
                all(m[k].state == StateEnum.blocked for k in corners):
            return corners
    return None


def _fill_open_seats(m, log=None, interior=None):
    """Fill every seat of the state-only map m in place - a free cell with
    the three other corners of one of its 2x2 blocks blocked - by choosing
    it (its free diagonal neighbours blocked), to a fixed point. Returns the
    cells filled. log: optional list; gets one dict per round - the seats
    found (with the blocked corners that make each a seat), and per seat
    either the cells its choice blocked or, if an earlier seat of the same
    round blocked it first, which one (2026-11-13, for the open-seats page)."""
    rows, cols = m.shape
    filled = []
    while True:
        seats = [(x, _seat_block(m, x, interior)) for x in np.ndindex(m.shape) if m[x].state == StateEnum.free]
        seats = [(x, corners) for x, corners in seats if corners]
        if not seats:
            return filled
        round_log = []
        blocked_by = {}
        for x, corners in seats:
            if m[x].state != StateEnum.free:
                round_log.append({"seat": x, "corners": corners, "lost_to": blocked_by.get(x)})
                continue
            m[x].state = StateEnum.chosen
            filled.append(x)
            newly = []
            for a, b in _DIAGONALS:
                y = (x[0] + a, x[1] + b)
                if 0 <= y[0] < rows and 0 <= y[1] < cols and m[y].state == StateEnum.free:
                    m[y].state = StateEnum.blocked
                    blocked_by[y] = x
                    newly.append(y)
            round_log.append({"seat": x, "corners": corners, "blocks": newly})
        if log is not None:
            log.append(round_log)


def _holes(states, shape, interior=None):
    """Upper-left corners of the (real, see _real_block) 2x2 blocks with no
    free or chosen cell - nothing can ever cover them."""
    rows, cols = shape
    return [(r, k) for r in range(rows - 1) for k in range(cols - 1)
            if _real_block(shape, interior, r, k)
            and all(states[(r + a, k + b)] == StateEnum.blocked for a in (0, 1) for b in (0, 1))]


def _check_against_simulation(mem, states, shape):
    """Every free candidate: if its claims contain two cells that block each
    other, choosing it must really leave an uncoverable 2x2 block (a
    contradiction the suspension steps did not catch); otherwise its claims
    must equal the simulation exactly. Returns the contradictory
    candidates."""
    interior = getattr(mem, "interior", None)
    contradictory = []
    for c in np.ndindex(shape):
        if not (Pyramid.is_candidate(c) and mem.map[c].state == StateEnum.free):
            continue
        schema = candidate_schema(mem, c)
        claimed = {tuple(int(v) for v in k.split(",")) for k in schema["forced"]}
        forced, final = _simulate_forced(states, shape, c, return_states=True, interior=interior)
        if _contradiction(schema):
            assert _holes(final, shape, interior), c
            contradictory.append(c)
        else:
            assert claimed == forced, c
    return contradictory


def _run(raw, tiles, size, origin, monkeypatch, clip=False, interior=None):
    """One pass (clip, interior: see cores.DeviceMemory), checked: fixed point
    of the round trips, every candidate against the simulation."""
    Pyramid.set_placement_tiles(tiles, size, origin)
    states = {idx: raw[idx].state for idx in np.ndindex(raw.shape)}
    assert _holes(states, raw.shape, interior) == []
    mem = DeviceMemory(_state_only(raw), size, origin=origin, clip=clip, interior=interior)
    entries = _onepass_recording_entries(mem, monkeypatch)
    _check_entries(mem, entries)
    assert _nary_in_isolation(mem, entries) == []   # the round trips reached a fixed point
    # the pass is analysis only (2026-11-16): no .state changed. Every
    # candidate it reports refuted (mem.refuted, suspended in the pass) is
    # really impossible - choosing it leaves a hole
    assert {idx: mem.map[idx].state for idx in np.ndindex(raw.shape)} == states
    assert set(mem.refuted) == mem.suspended
    for c in mem.refuted:
        _forced, final = _simulate_forced(states, raw.shape, c, return_states=True, interior=interior)
        assert _holes(final, raw.shape, interior), c
    contradictory = _check_against_simulation(mem, states, raw.shape)
    return mem, entries, contradictory


def _write_pages(mem, entries, tiles, name, title, note):
    os.makedirs(SUPER_DIR, exist_ok=True)
    render_links_html(mem, os.path.join(SUPER_DIR, name + ".html"), title, entries, note)
    render_candidate_maps_html(mem, os.path.join(SUPER_DIR, name + "_candidates.html"),
                               title + ": every candidate", sorted(tiles),
                               note + " One map per free candidate of every placement tile - what it "
                               "forces on its own; its tile is outlined bold.")


def completable_swept_map(n, M, seed):
    """build_swept_map's board, reduced to .state, its open seats filled -
    or None if that leaves a 2x2 block nothing can cover: the sweep places
    without checking, and such a board has no valid completion, so it is
    discarded (2026-11-13). Returns (raw, filled seats) or None."""
    m, N = build_swept_map(n, M, seed=seed)
    raw = _state_only(m)
    filled = _fill_open_seats(raw)
    if _holes({idx: raw[idx].state for idx in np.ndindex(raw.shape)}, raw.shape):
        return None
    return raw, filled


def test_superlattice_swept_maps(monkeypatch):
    """The swept maps with the odd-odd superlattice phase as placement tiles
    (9 tiles), open seats filled first; uncompletable boards discarded."""
    discarded = []
    for seed in SWEPT_SEEDS:
        board = completable_swept_map(7, 3, seed)
        if board is None:
            discarded.append(seed)
            continue
        raw, filled = board
        tiles = superlattice_tiles(7, 7, (1, 1))
        mem, entries, contradictory = _run(raw, tiles, 3, 0, monkeypatch)
        assert len(tiles) == 9
        # suspended after the n-ary step (2026-11-13, Base.contradictory_after_nary)
        assert contradictory == []
        late = {1: {(9, 11)}, 2: set()}[seed]
        assert late <= mem.suspended
        _write_pages(mem, entries, tiles, f"swept_seed{seed}",
                     f"Swept map seed {seed}, superlattice placement tiles",
                     f"build_swept_map(n=7, M=3, seed={seed}), reduced to .state; its {len(filled)} open "
                     "seats filled first; placement tiles (1+2I, 1+2J) - 9 tiles, all candidates in one "
                     "pass; Base.init_map and one pass."
                     + (f" Suspended only after the n-ary step (their n-ary facts make them "
                        f"force two cells that block each other; the pass was rerun): "
                        f"{', '.join(map(str, sorted(late)))}." if late else ""))
    # seed 0: filling its open seats is a cascade of 18 seats in 10 rounds;
    # in round 8, seats (8,13) and (11,14) block (9,12) and (10,13), the
    # last free cells of the 2x2 block at (9,12) (test_swept_maps_open_seats_page)
    assert discarded == [0]


def test_superlattice_frozen_area(monkeypatch):
    """The frozen area with the even-even superlattice phase as placement
    tiles: (0,0), (0,2), (2,0), (2,2) - four placement tiles at once."""
    raw = _state_only(_frozen_area_map())
    assert _fill_open_seats(raw) == []
    tiles = superlattice_tiles(3, 3, (0, 0))
    mem, entries, contradictory = _run(raw, tiles, 3, 2, monkeypatch)
    assert contradictory == []
    _write_pages(mem, entries, tiles, "frozen_area",
                 "Frozen area, superlattice placement tiles",
                 "test_sudden_appearance.test_frozen_area's board after round 2, reduced to .state; "
                 "placement tiles (0,0), (0,2), (2,0), (2,2) (origin 2) - four tiles, all candidates "
                 "in one pass; Base.init_map and one pass.")


def _grid(m):
    code = {StateEnum.free: 0, StateEnum.chosen: 1, StateEnum.blocked: 2}
    return [[code[m[r, k].state] for k in range(m.shape[1])] for r in range(m.shape[0])]


def test_swept_maps_open_seats_page():
    """The swept maps carry seats the sweep never filled (Ivo, 2026-11-13:
    "The swept maps had seats that were never filled. show it"): one page,
    per seed the board as build_swept_map leaves it with its open seats,
    and the board after filling them round by round - for seed 0 with the
    uncoverable 2x2 block at (9,12)."""
    maps = []
    for seed in SWEPT_SEEDS:
        m, N = build_swept_map(7, 3, seed=seed)
        raw = _state_only(m)
        before = _grid(raw)
        log = []
        filled = _fill_open_seats(raw, log)
        holes = _holes({idx: raw[idx].state for idx in np.ndindex(raw.shape)}, raw.shape)
        assert len(log[0]) == {0: 4, 1: 9, 2: 6}[seed]          # open seats as given
        assert len(filled) == {0: 18, 1: 19, 2: 8}[seed]
        assert holes == ([(9, 12)] if seed == 0 else [])
        maps.append({"name": f"Swept map, seed {seed}", "rows": N, "cols": N, "tile_size": 3, "origin": 0,
                     "before": before, "after": _grid(raw), "rounds": log, "holes": holes})
    os.makedirs(SUPER_DIR, exist_ok=True)
    render_open_seats_html(os.path.join(SUPER_DIR, "swept_open_seats.html"), "Swept maps: unfilled seats",
                           maps, "build_swept_map(n=7, M=3, seed) places one cell per core in five phases "
                           "but never fills the seats those placements create - free cells whose 2x2 block "
                           "has its three other corners blocked, so they must be chosen. Left: the board as "
                           "built, its open seats and the blocked corners that make them seats. Right: the "
                           "board after choosing every seat (blocking its free diagonal neighbours) until "
                           "none is left; filling a seat can create new ones, so it runs in rounds. Red in "
                           "the round list: a cell of an uncoverable 2x2 block.")


# --- .blocked lists (2026-11-14) -------------------------------------------------------

def _conflict_when_both_chosen(states, shape, pair):
    """Choose both cells of pair (the second only if the first left it
    free), fill the seats to a fixed point: True if that fails - the second
    got blocked, or a 2x2 block is left that nothing can cover."""
    rows, cols = shape
    st = dict(states)

    def choose(x):
        st[x] = StateEnum.chosen
        for a, b in _DIAGONALS:
            y = (x[0] + a, x[1] + b)
            if 0 <= y[0] < rows and 0 <= y[1] < cols and st[y] == StateEnum.free:
                st[y] = StateEnum.blocked

    def blocked(r, k):
        return 0 <= r < rows and 0 <= k < cols and st[(r, k)] == StateEnum.blocked

    for c in pair:
        if st[c] != StateEnum.free:
            return True
        choose(c)
    while True:
        seats = [x for x, v in st.items() if v == StateEnum.free and any(
            all(blocked(r, k) for r, k in ((x[0] + a, x[1]), (x[0], x[1] + b), (x[0] + a, x[1] + b)))
            for a, b in _DIAGONALS)]
        if not seats:
            return bool(_holes(st, shape))
        for x in seats:
            if st[x] == StateEnum.free:
                choose(x)


def _check_blocked(mem, states, shape, origin):
    """After a pass: the base tiles' .blocked lists against the simulation,
    the supertiles' as the union of their children's. Returns (pairs,
    self-blocking)."""
    base = set().union(*(t.blocked for t in mem.pyramid[0].flat))
    for level in range(1, len(mem.pyramid)):
        for core in mem.pyramid[level].flat:
            assert core.blocked == set().union(*(c.blocked for c in core.elements.flat))
    assert mem.pyramid[-1][0, 0].blocked == base
    pairs = {p for p in base if p[0] != p[1]}
    selfs = {p[0] for p in base if p[0] == p[1]}
    assert selfs == mem.suspended                  # self-blocking = suspended
    for pair in pairs:                             # every pair listed is a real conflict
        assert _conflict_when_both_chosen(states, shape, pair), pair
    # every real conflict NOT listed comes only from choosing both together
    # (new seats neither makes alone - compounds, not computed yet)
    cands = [c for c in np.ndindex(shape) if Pyramid.is_candidate(c) and mem.map[c].state == StateEnum.free]

    def tile(c):
        return ((c[0] - origin) // 3, (c[1] - origin) // 3)

    alone = {c: _simulate_forced(states, shape, c) | {c} for c in cands}
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if tile(a) == tile(b) or (a, b) in pairs or not _conflict_when_both_chosen(states, shape, (a, b)):
                continue
            assert not any((x[0] + 1, x[1] + d) in alone[b] or (x[0] - 1, x[1] + d) in alone[b]
                           for x in alone[a] for d in (-1, 1)), (a, b)
    return pairs, selfs


def test_blocked_lists(monkeypatch):
    """Ivo, 2026-11-14: "Its time to re-introduce keeping tab of blocking and
    self-blocking but we dont do unblocking yet. So each time we touch base
    and done everything else, fill the tile's .blocked list and also update
    the .blocked list of supertiles." (Pyramid.update_blocked, run when the
    round trips of a step reach their fixed point.) On the superlattice
    maps: every listed pair is a real conflict, the self-blocking entries
    are the suspended candidates, supertiles hold the union of their
    children's lists, and every unlisted conflict needs both candidates
    chosen together (a compound)."""
    raw = _state_only(_frozen_area_map())
    tiles = superlattice_tiles(3, 3, (0, 0))
    mem, _entries, _c = _run(raw, tiles, 3, 2, monkeypatch)
    states = {idx: raw[idx].state for idx in np.ndindex(raw.shape)}
    pairs, selfs = _check_blocked(mem, states, raw.shape, 2)
    assert len(pairs) == 30 and selfs == set()
    for seed, n_pairs in ((1, 4), (2, 0)):
        raw, _filled = completable_swept_map(7, 3, seed)
        mem, _entries, _c = _run(raw, superlattice_tiles(7, 7, (1, 1)), 3, 0, monkeypatch)
        states = {idx: raw[idx].state for idx in np.ndindex(raw.shape)}
        pairs, selfs = _check_blocked(mem, states, raw.shape, 0)
        assert len(pairs) == n_pairs and selfs == mem.suspended


# --- entries per tile, coloured by candidate (2026-11-14) -------------------------------

ENTRIES_DIR = os.path.join(SUPER_DIR, "entries")


def test_entries_pages(monkeypatch):
    """Ivo, 2026-11-14: "The current display is for checking correctness. But
    as I said the result we are interested in as the entries for each tile
    from (possibly far-away) candidates. make a new subfolder in
    Superlattice and new htlms. There, you give each placement tile a
    primary color, and each candidate therein a hue of that primary color.
    Then show id_ids from candiatate as dots of that color. If an id_in is
    reachable from several candidates just shift the dots a bit. Nothing
    else." One page per superlattice map (links_html.render_entries_html)."""
    os.makedirs(ENTRIES_DIR, exist_ok=True)
    note = ("Where each candidate's chain enters a tile, after one pass: a cell of a tile that the "
            "candidate forces - through a single link from outside the tile, or as an n-ary fact - "
            "or the candidate itself. One colour per placement tile, one shade per candidate; "
            "several candidates entering the same cell are shifted apart. Hover or tap a candidate "
            "in the legend to see only its dots.")
    maps = [("frozen_area", "Frozen area: entries per tile", _state_only(_frozen_area_map()),
             superlattice_tiles(3, 3, (0, 0)), 2)]
    for seed in (1, 2):
        raw, _filled = completable_swept_map(7, 3, seed)
        maps.append((f"swept_seed{seed}", f"Swept map seed {seed}: entries per tile", raw,
                     superlattice_tiles(7, 7, (1, 1)), 0))
    for name, title, raw, tiles, origin in maps:
        mem, _entries, _c = _run(raw, tiles, 3, origin, monkeypatch)
        entries = candidate_entries(mem)
        for c, cells in entries.items():
            assert c in cells                                     # its own tile
            assert cells <= Base.claimed(mem, c) | {c}            # only cells it forces
        render_entries_html(mem, os.path.join(ENTRIES_DIR, name + ".html"), title, note)
