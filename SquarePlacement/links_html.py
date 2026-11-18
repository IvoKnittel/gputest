"""Render a map with its links as an HTML page, in the schema of
upstream-forced-by.html (project root) - added 2026-11-10 for testing n-ary
link generation in isolation (Ivo: "Write a function that translates a map
with links onto a schema like in upstream-forced-by.html and apply it in the
test for display. Also show the map with state and links.").

render_links_html(mem, path, title, entries=None) reads a cores.DeviceMemory:
- mem.map: every cell's .state, and its conditions (SquareItem.forced_by) -
  size-1 conditions are drawn as single links (forcer -> cell), size-2
  conditions as pair conditions (both members -> cell, a checkbox shows
  them), size-3 ones only counted;
- the global placement-tile layout (cores.Pyramid.is_candidate) and
  mem.tile_size / mem.origin: placement tiles shaded, tile borders drawn;
- mem.pyramid[0], if built: each base tile's .links triples - without the
  dead ends (id_in, DUMMY, DUMMY), which are not displayed - drawn on the
  map; single links are shown only where they lie on such a triple's chain
  (live_singles), and only if no other sequence of links connects the same
  two cells (without_redundant) (fat rings at id_out, filled dots at id_in, small
  dots at id_exit);
- entries (optional): each base tile's simultaneous entries - a dict
  {(ci, cj): set of (candidate, cell)} or an array laid out like
  mem.pyramid[0] - kept in the page data (map_to_schema), not displayed.

The page is self-contained (data embedded as JSON, drawing in inline JS),
light and dark theme, nothing loaded at runtime except Google Fonts.
"""
import html
import json

import numpy as np

from cores import Pyramid, Base, DUMMY
import cores
from map_of_squares import StateEnum

_STATE = {StateEnum.free: 0, StateEnum.chosen: 1, StateEnum.blocked: 2}


def _key(id_):
    return id_ if id_ == DUMMY else f"{id_[0]},{id_[1]}"


def live_singles(mem):
    """The single links that lie on the chain of some non-dead-end triple of
    a base tile (2026-11-13, Ivo: "remove also the links belonging to
    dead-end items"): for each (id_in, id_out, id_exit) of a base tile's
    .links with id_out not DUMMY, the links on a path from id_in to id_out
    inside the tile, plus the exit link id_out -> id_exit. Links of chains
    that die, and links on no triple's chain at all, are left out. Returns
    a set of (forcer, cell) pairs."""
    live = set()
    for tile in mem.pyramid[0].flat:
        interior, forward, _exits, _status = Base.edges(tile)
        backward = {}
        for u, vs in forward.items():
            for v in vs & interior:
                backward.setdefault(v, set()).add(u)
        for a, out, exit_id in tile.links:
            if out == DUMMY:
                continue
            reached = {a} if a in interior else forward.get(a, set()) & interior
            frontier = list(reached)
            while frontier:
                u = frontier.pop()
                for v in forward.get(u, ()):
                    if v in interior and v not in reached:
                        reached.add(v)
                        frontier.append(v)
            to_out, frontier = {out}, [out]
            while frontier:
                v = frontier.pop()
                for u in backward.get(v, ()):
                    if u in interior and u not in to_out:
                        to_out.add(u)
                        frontier.append(u)
            for u in reached | {a}:
                for v in forward.get(u, ()):
                    if v in reached and v in to_out:
                        live.add((u, v))
            live.add((out, exit_id))
    return live


def without_redundant(edges, remove_first=frozenset()):
    """Remove every link a -> b for which another sequence of links from a to
    b exists (2026-11-13, Ivo: "Every link is redundant for which a sequence
    of links exists and can be removed") - one at a time, each checked
    against the links still left, so what reaches what never changes. This
    also drops links between cells that are not neighbours whenever a chain
    connects them. remove_first: links to try removing before all others -
    where two paths explain the same reach, the ones in remove_first are the
    ones that go. Returns the remaining set."""
    out = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
    for a, b in sorted(edges, key=lambda e: (e not in remove_first, e)):
        out[a].discard(b)
        seen, frontier = {a}, [a]
        found = False
        while frontier and not found:
            u = frontier.pop()
            for v in out.get(u, ()):
                if v == b:
                    found = True
                    break
                if v not in seen:
                    seen.add(v)
                    frontier.append(v)
        if not found:
            out[a].add(b)
    return {(a, b) for a, bs in out.items() for b in bs}


def map_to_schema(mem, entries=None):
    """The data the page draws, as plain JSON-able Python - also usable on
    its own (e.g. to compare two maps)."""
    m = mem.map
    rows, cols = m.shape
    states = [[_STATE[m[i, j].state] for j in range(cols)] for i in range(rows)]
    candidates = [_key(idx) for idx in np.ndindex(m.shape) if Pyramid.is_candidate(idx)]
    singles, pairs, triple_count = [], [], {}
    for idx in np.ndindex(m.shape):
        for condition in m[idx].forced_by:
            members = sorted(pos for pos, _status in condition)
            if len(members) == 1:
                singles.append([_key(members[0]), _key(idx)])
            elif len(members) == 2:
                pairs.append([_key(members[0]), _key(members[1]), _key(idx)])
            else:
                triple_count[_key(idx)] = triple_count.get(_key(idx), 0) + 1

    n_all_singles = len(singles)
    if mem.pyramid:
        shown = without_redundant(live_singles(mem))
    else:
        shown = without_redundant({(tuple(map(int, a.split(","))), tuple(map(int, b.split(","))))
                                   for a, b in singles})
    singles = [[_key(a), _key(b)] for a, b in shown]

    tiles = {}
    if mem.pyramid:
        base = mem.pyramid[0]
        for (ci, cj), tile in np.ndenumerate(base):
            if isinstance(entries, dict):
                tile_entries = entries.get((ci, cj), set())
            elif entries is not None:
                tile_entries = entries[ci, cj]
            else:
                tile_entries = set()
            tiles[f"{ci},{cj}"] = {
                "realm": [list(tile.realm[0]), list(tile.realm[1])],
                # dead ends (id_in, DUMMY, DUMMY) left out - irrelevant at this stage
                # (Ivo, 2026-11-13); the data itself still has them
                "links": sorted([_key(a), _key(b), _key(c)] for a, b, c in tile.links if b != DUMMY),
                "entries": sorted([_key(a), _key(b)] for a, b in tile_entries),
            }
    return {
        "rows": rows, "cols": cols,
        "tile_size": mem.tile_size, "origin": mem.origin,
        "states": states,
        "candidates": candidates,
        "suspended": sorted(_key(x) for x in getattr(mem, "suspended", ())),
        "singles": sorted(singles),
        "n_all_singles": n_all_singles,
        "pairs": sorted(pairs),
        "triple_count": triple_count,
        "tiles": tiles,
    }


def render_links_html(mem, path, title, entries=None, note=""):
    """Write the page for mem (see the module docstring) to `path`. note:
    an optional sentence shown under the title."""
    data = map_to_schema(mem, entries)
    page = _TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTE__", html.escape(note))
    page = page.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: the map on the left, the base tiles' lists on the right; one column on narrow screens. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1b2230; --muted: #5d6676; --rule: #d5dae2;
  --grid: #e6e9ee; --tile: #2a3346; --place: #fff1d6;
  --chosen: #1b2230; --blocked: #c9ced8; --link: #2563c9; --pair: #7a3fc4;
  --in: #178a55; --out: #c9580f; --exit: #c2352b; --cand: #c9580f;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
    --grid: #252c37; --tile: #c3cad6; --place: #3a2f17;
    --chosen: #e3e7ee; --blocked: #3b4350; --link: #6ea0ff; --pair: #b48bf0;
    --in: #45c88a; --out: #f08a45; --exit: #ff7a6e; --cand: #f08a45; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
  --grid: #252c37; --tile: #c3cad6; --place: #3a2f17;
  --chosen: #e3e7ee; --blocked: #3b4350; --link: #6ea0ff; --pair: #b48bf0;
  --in: #45c88a; --out: #f08a45; --exit: #ff7a6e; --cand: #f08a45; color-scheme: dark;
}
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1400px; margin: 0 auto; padding-block: 24px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-family: var(--font-head); font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }
h2 { font-family: var(--font-head); font-size: 1.05rem; margin: 0 0 8px; }
.lede { color: var(--muted); max-width: 80ch; margin: 0 0 14px; }
.stats { font-family: var(--font-mono); font-size: .82rem; color: var(--muted); margin: 0 0 14px; }

.figure { display: grid; gap: 10px; }
.svgbox { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow: auto; max-height: 86vh; }
svg { display: block; }
svg text { font-family: var(--font-mono); }
.controls { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .88rem; }
.controls label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: .82rem; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 16px; }
.label { font: 600 .66rem var(--font-body); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-top: 4px; }
.item { font-family: var(--font-mono); font-size: .76rem; overflow-wrap: anywhere; }
.empty { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); }
.cnt { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
</style>

<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__NOTE__</p>
  <p class="stats" id="stats"></p>
  <div class="figure">
      <div class="controls">
        <label><input type="checkbox" id="show-singles" checked> single links</label>
        <label><input type="checkbox" id="show-pairs"> pair conditions</label>
        <label><input type="checkbox" id="show-tilelinks" checked> tile links (in / out / exit)</label>
      </div>
      <div class="svgbox"><svg id="board" role="img" aria-label="Map with cell states, tiles and links"></svg></div>
      <div class="legend">
        <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--chosen)"/></svg>chosen</span>
        <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--blocked)"/></svg>blocked</span>
        <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--place)" stroke="var(--rule)"/></svg>eligible cells</span>
        <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="#ffffff" stroke="var(--rule)"/></svg>suspended candidate</span>
        <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="none" stroke="var(--cand)" stroke-width="2" stroke-dasharray="3 2"/></svg>free candidate</span>
        <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--link)" stroke-width="2"/></svg>single link</span>
        <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--pair)" stroke-width="1.5" stroke-dasharray="3 2"/></svg>pair condition</span>
        <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="5" fill="var(--in)"/></svg>id_in</span>
        <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="5" fill="none" stroke="var(--out)" stroke-width="3"/></svg>id_out</span>
        <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="3" fill="var(--exit)"/></svg>id_exit</span>
      </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("board");
const R = DATA.rows, C = DATA.cols;
const S = Math.max(16, Math.min(40, Math.floor(820 / Math.max(R, C))));
const O = 26;
const W = O + C * S + 8, H = O + R * S + 8;
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
svg.setAttribute("width", W); svg.setAttribute("height", H);
const cx = c => O + c * S + S / 2, cy = r => O + r * S + S / 2;
const parse = k => k.split(",").map(Number);
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
const isCand = new Set(DATA.candidates);
const isSus = new Set(DATA.suspended || []);
const ts = DATA.tile_size, org = DATA.origin;
const tileOf = k => { const [r, c] = parse(k); if (r < org || c < org) return null; return Math.floor((r - org) / ts) + "," + Math.floor((c - org) / ts); };

const defs = el("defs", {});
for (const [id, col] of [["l", "--link"], ["p", "--pair"]]) {
  const mk = el("marker", { id: "a" + id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0 0 L10 5 L0 10 z", fill: `var(${col})` }, mk);
}

/* placement tiles (shaded) */
for (const k in DATA.tiles) {
  const t = DATA.tiles[k], [[r0, r1], [c0, c1]] = t.realm;
  if (isCand.has(r0 + "," + c0)) el("rect", { x: O + c0 * S, y: O + r0 * S, width: (c1 - c0) * S, height: (r1 - r0) * S, fill: "var(--place)" });
}
/* cells */
for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
  const st = DATA.states[r][c];
  const sus = st === 0 && isSus.has(r + "," + c);
  el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill: st === 1 ? "var(--chosen)" : st === 2 ? "var(--blocked)" : sus ? "#ffffff" : "none", stroke: "var(--grid)", "stroke-width": 1 });
  if (st === 0 && isCand.has(r + "," + c)) el("rect", { x: O + c * S + 3, y: O + r * S + 3, width: S - 6, height: S - 6, fill: "none", stroke: "var(--cand)", "stroke-width": 1.3, "stroke-dasharray": "3 2" });
}
/* tile borders */
for (const k in DATA.tiles) {
  const [[r0, r1], [c0, c1]] = DATA.tiles[k].realm;
  el("rect", { x: O + c0 * S, y: O + r0 * S, width: (c1 - c0) * S, height: (r1 - r0) * S, fill: "none", stroke: "var(--tile)", "stroke-width": 1.8 });
  el("text", { x: O + c0 * S + 3, y: O + r0 * S + 10, "font-size": 9, fill: "var(--muted)" }).textContent = "(" + k + ")";
}
/* axis labels */
const step = S < 22 ? 2 : 1;
for (let c = 0; c < C; c += step) el("text", { x: cx(c), y: O - 8, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" }).textContent = c;
for (let r = 0; r < R; r += step) el("text", { x: O - 6, y: cy(r) + 3, "text-anchor": "end", "font-size": 10, fill: "var(--muted)" }).textContent = r;

const arrow = (a, b, col, marker, width, dash, parent) => {
  const [r1, c1] = parse(a), [r2, c2] = parse(b);
  const x1 = cx(c1), y1 = cy(r1), x2 = cx(c2), y2 = cy(r2);
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, sh = S * 0.28;
  el("line", { x1: x1 + dx / len * sh, y1: y1 + dy / len * sh, x2: x2 - dx / len * sh, y2: y2 - dy / len * sh,
    stroke: col, "stroke-width": width, "stroke-dasharray": dash || "none", "marker-end": marker ? `url(#${marker})` : "none" }, parent);
};
const gPairs = el("g", {}), gSingles = el("g", {}), gTiles = el("g", {});
for (const [a, b, cell] of DATA.pairs) {
  arrow(a, cell, "var(--pair)", "ap", 1.2, "3 2", gPairs);
  arrow(b, cell, "var(--pair)", "ap", 1.2, "3 2", gPairs);
}
for (const [a, b] of DATA.singles) arrow(a, b, "var(--link)", "al", 1.8, null, gSingles);

function drawTileLinks(only) {
  gTiles.replaceChildren();
  const ins = new Set(), outs = new Set(), exits = new Set();
  for (const k in DATA.tiles) {
    if (only && k !== only) continue;
    for (const [a, b, c] of DATA.tiles[k].links) {
      if (a !== "dummy") ins.add(a);
      if (b !== "dummy") outs.add(b);
      if (c !== "dummy") exits.add(c);
    }
  }
  const inside = k => { const [r, c] = parse(k); return r >= 0 && c >= 0 && r < R && c < C; };
  for (const k of ins) if (inside(k)) { const [r, c] = parse(k); el("circle", { cx: cx(c), cy: cy(r), r: S * 0.2, fill: "var(--in)" }, gTiles); }
  for (const k of outs) if (inside(k)) { const [r, c] = parse(k); el("circle", { cx: cx(c), cy: cy(r), r: S * 0.27, fill: "none", stroke: "var(--out)", "stroke-width": Math.max(2, S * 0.09) }, gTiles); }
  for (const k of exits) if (inside(k)) { const [r, c] = parse(k); el("circle", { cx: cx(c), cy: cy(r), r: S * 0.11, fill: "var(--exit)" }, gTiles); }
}
const vis = () => {
  gSingles.style.display = document.getElementById("show-singles").checked ? "" : "none";
  gPairs.style.display = document.getElementById("show-pairs").checked ? "" : "none";
  gTiles.style.display = document.getElementById("show-tilelinks").checked ? "" : "none";
};
for (const id of ["show-singles", "show-pairs", "show-tilelinks"]) document.getElementById(id).addEventListener("change", vis);

const keys = Object.keys(DATA.tiles);
const nTripleCells = Object.keys(DATA.triple_count).length;
document.getElementById("stats").textContent =
  `${R}×${C} cells · ${keys.length} base tiles of ${ts}×${ts} from origin ${org} · ` +
  `${DATA.singles.length} single links shown of ${DATA.n_all_singles} (dead ends and redundant ones left out) · ` +
  `${DATA.pairs.length} pair conditions · ${nTripleCells} cells with triple conditions`;
drawTileLinks(null); vis();
</script>
</html>
"""


# --- one map per candidate (2026-11-12) ----------------------------------------------

def _prefer_own_tile(edges, keep, candidate, forced, size, origin):
    """Drop the arrows into a cell from other tiles once a cell of its own
    tile forces it too (2026-11-13, Ivo: "remove the arrow (2,3) to (1,5)
    because that cell is already forced by a cell of the same tile"). An
    arrow goes only if every forced cell stays reachable from the candidate
    without it (no cycle may lose its way in); arrows in `keep` (the joint
    arrows of a collapsed condition, drawn as a group) are never dropped.
    Cells outside every tile (the border, before `origin`) have no tile."""
    def tile_of(x):
        if x[0] < origin or x[1] < origin:
            return None
        return ((x[0] - origin) // size, (x[1] - origin) // size)

    def reaches_all(es):
        out = {}
        for a, b in es:
            out.setdefault(a, set()).add(b)
        seen, frontier = {candidate}, [candidate]
        while frontier:
            for v in out.get(frontier.pop(), ()):
                if v not in seen:
                    seen.add(v)
                    frontier.append(v)
        return forced <= seen

    edges = set(edges)
    for a, b in sorted(edges):
        tb = tile_of(b)
        if (a, b) in keep or tb is None or tile_of(a) == tb:
            continue
        if not any(x != a and y == b and tile_of(x) == tb for x, y in edges):
            continue
        if reaches_all(edges - {(a, b)}):
            edges.discard((a, b))
    return edges


def candidate_schema(mem, candidate):
    """What one candidate does in mem after a pass, as plain JSON-able data:
    the cells it forces through single links and its n-ary collapse targets
    (mem.collapsed) - the closure from the candidate -, the links of that
    chain, the base-tile triples that start at it and whose exit link is a
    drawn chain arrow, and the conditions with two or more members that name it
    (reduced conditions left by partial collapses, or original ones)."""
    m = mem.map
    forced, frontier = set(), {candidate}
    edges = set()
    # since 2026-11-13 a full n-ary collapse writes no map link (only the tile
    # holding y learns it), so the candidate's collapse targets come from
    # mem.collapsed: (candidate, y) means y is forced when candidate is chosen
    collapsed_targets = {y for (c, y) in getattr(mem, "collapsed", {}) if c == candidate}
    edges |= {(candidate, y) for y in collapsed_targets}
    forced |= collapsed_targets
    frontier |= collapsed_targets
    while frontier:
        nxt = set()
        for idx in np.ndindex(m.shape):
            if idx == candidate:
                continue
            for condition in m[idx].forced_by:
                if len(condition) == 1:
                    (a, _s), = condition
                    if a in frontier:
                        edges.add((a, idx))
                        if idx not in forced:
                            nxt.add(idx)
        forced |= nxt
        frontier = nxt
    # a link the n-ary step made by collapsing a condition (mem.collapsed) is
    # shown through that condition's members - the cells that together
    # force it - instead of as a long arrow from the forcer (2026-11-13).
    # Redundant links go first (collapsed ones included): only a collapsed
    # link that is the actual reason a cell is forced gets its members drawn,
    # and those arrows stay together as a group.
    collapsed = getattr(mem, "collapsed", {})
    joint_edges = set()
    shown = set()
    # links back INTO the candidate are not drawn (it is the start), but they
    # are real paths, so they count when deciding what is redundant
    into = {(x, candidate) for x in forced
            if any(len(c) == 1 and next(iter(c))[0] == x for c in m[candidate].forced_by)}
    # where a plain chain explains the same reach, the collapsed link goes
    is_collapsed = {e for e in edges if e in collapsed}
    for a, b in without_redundant(edges | into, remove_first=is_collapsed) - into:
        members = None
        for condition in sorted(collapsed.get((a, b), ()), key=sorted):
            if condition <= forced | {candidate}:
                members = condition
                break
        if members is None:
            shown.add((a, b))
        else:
            joint_edges |= {(x, b) for x in members if x != b}
    shown |= joint_edges
    shown = _prefer_own_tile(shown, joint_edges, candidate, forced, mem.tile_size, mem.origin)
    # only triples whose exit link id_out -> id_exit is an arrow of the drawn
    # chain (2026-11-13, Ivo: option a) - the exit layer marks which chain
    # arrows cross a tile border; an exit link the redundancy removal
    # dropped would show up as a stray second path
    triples = []
    if mem.pyramid:
        for (ci, cj), tile in np.ndenumerate(mem.pyramid[0]):
            triples += [[f"{ci},{cj}", _key(a), _key(b), _key(c)] for a, b, c in tile.links
                        if a == candidate and b != DUMMY and (b, c) in shown]
    multi = []
    for idx in np.ndindex(m.shape):
        for condition in m[idx].forced_by:
            members = sorted(pos for pos, _s in condition)
            if len(members) >= 2 and candidate in members:
                multi.append([[_key(p) for p in members if p != candidate], _key(idx)])
    return {
        "candidate": _key(candidate),
        "forced": sorted(_key(x) for x in forced),
        "edges": sorted([_key(a), _key(b)] for a, b in shown if (a, b) not in joint_edges),
        "joint": sorted([_key(a), _key(b)] for a, b in shown if (a, b) in joint_edges),
        "triples": sorted(triples),
        "multi": sorted(multi),
    }


def _contradiction(schema):
    """Two diagonal neighbours among the candidate and the cells it forces
    (candidate_schema) - they block each other, so the candidate can never
    validly be chosen - or None. Pyramid.onepass suspends such candidates
    (Base.suspend_contradictory, Base.contradictory_after_nary), so after a
    pass this should never fire - the page notes it if it does."""
    cells = {tuple(int(v) for v in k.split(",")) for k in schema["forced"] + [schema["candidate"]]}
    for x in sorted(cells):
        for dc in (-1, 1):
            y = (x[0] + 1, x[1] + dc)
            if y in cells:
                return x, y
    return None


def render_candidate_maps_html(mem, path, title, tile, note=""):
    """One page with one map per free candidate of base tile `tile` (ci, cj)
    of mem.pyramid[0] (mem after a pass): see candidate_schema. Cells of the
    tile that are not free, or suspended for the pass, get no map and are
    named on the page; suspended cells are drawn white on every map.

    `tile` may also be a list of tiles (2026-11-13, superlattice: several
    placement tiles in one pass): then every map is labelled with its
    candidate's tile, and that tile is outlined bold on its map. A candidate
    whose forced cells contain two diagonal neighbours gets a note saying
    so (see _contradiction)."""
    base = map_to_schema(mem)
    tiles = [tile] if isinstance(tile[0], (int, np.integer)) else list(tile)
    several = len(tiles) > 1
    panels, other = [], []
    for t_idx in tiles:
        t = mem.pyramid[0][t_idx]
        (r0, r1), (c0, c1) = t.realm
        cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
        free = [x for x in cells if mem.map[x].state == StateEnum.free and Pyramid.is_candidate(x)]
        other += [[_key(x), "suspended" if x in mem.suspended else mem.map[x].state.name]
                  for x in cells if x not in free and (not several or x in mem.suspended)]
        for x in free:
            panel = candidate_schema(mem, x)
            panel["tile"] = f"{t_idx[0]},{t_idx[1]}"
            if several:
                panel["label"] = f"Candidate ({_key(x)}) · tile ({t_idx[0]},{t_idx[1]})"
            pair = _contradiction(panel)
            if pair:
                panel["note"] = (f"Contradictory: it forces ({_key(pair[0])}) and ({_key(pair[1])}), which "
                                 "block each other - yet not suspended.")
            panels.append(panel)
    data = {
        "map": {k: base[k] for k in ("rows", "cols", "tile_size", "origin", "states", "candidates", "suspended")},
        "tiles": {k: v["realm"] for k, v in base["tiles"].items()},
        "tile": "" if several else f"{tiles[0][0]},{tiles[0][1]}",
        "tile_label": (f"{len(tiles)} placement tiles" if several else ""),
        "other": other,
        "panels": panels,
    }
    page = _CANDIDATE_TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTE__", html.escape(note))
    page = page.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


_CANDIDATE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: one section per candidate - its name above its map. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1b2230; --muted: #5d6676; --rule: #d5dae2;
  --grid: #e6e9ee; --tile: #2a3346; --place: #fff1d6;
  --chosen: #1b2230; --blocked: #c9ced8; --forced: #f3c9a3; --link: #c9580f; --multi: #7a3fc4;
  --out: #2563c9; --exit: #c2352b; --cand: #c9580f; --new: #9fd6a8; --newlink: #178a55;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
    --grid: #252c37; --tile: #c3cad6; --place: #3a2f17;
    --chosen: #e3e7ee; --blocked: #3b4350; --forced: #6b3f1c; --link: #f08a45; --multi: #b48bf0;
    --out: #6ea0ff; --exit: #ff7a6e; --cand: #f08a45; --new: #1f5a2e; --newlink: #45c88a; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
  --grid: #252c37; --tile: #c3cad6; --place: #3a2f17;
  --chosen: #e3e7ee; --blocked: #3b4350; --forced: #6b3f1c; --link: #f08a45; --multi: #b48bf0;
  --out: #6ea0ff; --exit: #ff7a6e; --cand: #f08a45; --new: #1f5a2e; --newlink: #45c88a; color-scheme: dark;
}
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1400px; margin: 0 auto; padding-block: 24px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-family: var(--font-head); font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }
h2 { font-family: var(--font-head); font-size: 1.15rem; margin: 0 0 6px; }
.lede { color: var(--muted); max-width: 80ch; margin: 0 0 10px; }
.meta { font-family: var(--font-mono); font-size: .82rem; color: var(--muted); margin: 0 0 12px; }
.controls { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .88rem; margin: 0 0 8px; }
.controls label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: .82rem; margin: 0 0 18px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 16px; }
.toc { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 18px; }
.toc a { font-family: var(--font-mono); font-size: .82rem; color: var(--fg); background: var(--panel); border: 1px solid var(--rule); border-radius: 999px; padding: 2px 10px; text-decoration: none; }
.toc a:focus-visible { outline: 2px solid var(--out); outline-offset: 2px; }
section.cand { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; margin: 0 0 16px; }
.svgbox { overflow: auto; max-width: 100%; }
svg { display: block; }
svg text { font-family: var(--font-mono); }
.label { font: 600 .66rem var(--font-body); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-top: 8px; }
.items { font-family: var(--font-mono); font-size: .78rem; overflow-wrap: anywhere; }
.empty { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); }
.stepnote { color: var(--muted); font-size: .88rem; margin: 0 0 8px; }
</style>

<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__NOTE__</p>
  <p class="meta" id="meta"></p>
  <div class="controls">
    <label><input type="checkbox" id="show-multi"> conditions mentioning the candidate (pairs, triples)</label>
  </div>
  <div class="legend">
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--chosen)"/></svg>chosen</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--blocked)"/></svg>blocked</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--place)" stroke="var(--rule)"/></svg>eligible cells</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="#ffffff" stroke="var(--rule)"/></svg>suspended candidate</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--forced)"/></svg>forced by the candidate</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="none" stroke="var(--cand)" stroke-width="3"/></svg>the candidate</span>
    <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--link)" stroke-width="2"/></svg>single link of its chain</span>
    <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="5" fill="none" stroke="var(--out)" stroke-width="3"/></svg>id_out</span>
    <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="3" fill="var(--exit)"/></svg>id_exit</span>
    <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--out)" stroke-width="1.6" stroke-dasharray="5 2"/></svg>exit link id_out → id_exit</span>
    <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--multi)" stroke-width="1.5" stroke-dasharray="3 2"/></svg>conditions mentioning the candidate</span>
    <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--link)" stroke-width="2" stroke-dasharray="2 2"/></svg>compound forced</span>
    <span class="steps-only"><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--new)"/></svg>newly forced in this step</span>
    <span class="steps-only"><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--newlink)" stroke-width="3"/></svg>new link in this step</span>
  </div>
  <nav class="toc" id="toc" aria-label="Candidates"></nav>
  <div id="panels"></div>
</div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const M = DATA.map, R = M.rows, C = M.cols;
const S = Math.max(14, Math.min(30, Math.floor(640 / Math.max(R, C))));
const O = 24, W = O + C * S + 6, H = O + R * S + 6;
const cx = c => O + c * S + S / 2, cy = r => O + r * S + S / 2;
const parse = k => k.split(",").map(Number);
const fmt = k => k === "dummy" ? "dummy" : "(" + k + ")";
const isCand = new Set(M.candidates);
const isSus = new Set(M.suspended || []);
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
const inside = k => { if (k === "dummy") return false; const [r, c] = parse(k); return r >= 0 && c >= 0 && r < R && c < C; };
const multiGroups = [];

function arrow(g, a, b, col, width, dash, id) {
  const [r1, c1] = parse(a), [r2, c2] = parse(b);
  const x1 = cx(c1), y1 = cy(r1), x2 = cx(c2), y2 = cy(r2);
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, sh = S * 0.28;
  el("line", { x1: x1 + dx / len * sh, y1: y1 + dy / len * sh, x2: x2 - dx / len * sh, y2: y2 - dy / len * sh,
    stroke: col, "stroke-width": width, "stroke-dasharray": dash || "none", "marker-end": `url(#${id})` }, g);
}

function panel(p, n) {
  const sec = document.createElement("section");
  sec.className = "cand"; sec.id = "cand-" + n;
  const box = document.createElement("div"); box.className = "svgbox";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "What candidate " + fmt(p.candidate) + " forces");
  box.appendChild(svg); sec.appendChild(box);
  const defs = el("defs", {}, svg);
  for (const [id, col] of [["l" + n, "--link"], ["m" + n, "--multi"], ["n" + n, "--newlink"], ["o" + n, "--out"]]) {
    const mk = el("marker", { id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0 0 L10 5 L0 10 z", fill: `var(${col})` }, mk);
  }
  const forced = new Set(p.forced), fresh = new Set(p.new_forced || []);
  const freshEdges = new Set((p.new_edges || []).map(e => e.join("|")));
  for (const k in DATA.tiles) {
    const [[r0, r1], [c0, c1]] = DATA.tiles[k];
    if (isCand.has(r0 + "," + c0)) el("rect", { x: O + c0 * S, y: O + r0 * S, width: (c1 - c0) * S, height: (r1 - r0) * S, fill: "var(--place)" }, svg);
  }
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const st = M.states[r][c], k = r + "," + c;
    const fill = st === 1 ? "var(--chosen)" : st === 2 ? "var(--blocked)" : isSus.has(k) ? "#ffffff" : fresh.has(k) ? "var(--new)" : forced.has(k) ? "var(--forced)" : "none";
    el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill, stroke: "var(--grid)", "stroke-width": 1 }, svg);
  }
  for (const k in DATA.tiles) {
    const [[r0, r1], [c0, c1]] = DATA.tiles[k];
    el("rect", { x: O + c0 * S, y: O + r0 * S, width: (c1 - c0) * S, height: (r1 - r0) * S, fill: "none", stroke: "var(--tile)", "stroke-width": k === (p.tile || DATA.tile) ? 3 : 1.5 }, svg);
  }
  const step = S < 20 ? 2 : 1;
  for (let c = 0; c < C; c += step) el("text", { x: cx(c), y: O - 8, "text-anchor": "middle", "font-size": 9, fill: "var(--muted)" }, svg).textContent = c;
  for (let r = 0; r < R; r += step) el("text", { x: O - 5, y: cy(r) + 3, "text-anchor": "end", "font-size": 9, fill: "var(--muted)" }, svg).textContent = r;
  const [pr, pc] = parse(p.candidate);
  el("rect", { x: O + pc * S + 1.5, y: O + pr * S + 1.5, width: S - 3, height: S - 3, fill: "none", stroke: "var(--cand)", "stroke-width": 3 }, svg);
  const gm = el("g", {}, svg); multiGroups.push(gm);
  for (const [others, cell] of p.multi) {
    arrow(gm, p.candidate, cell, "var(--multi)", 1.1, "3 2", "m" + n);
    for (const o of others) if (inside(o)) arrow(gm, o, cell, "var(--multi)", 1.1, "3 2", "m" + n);
  }
  const ge = el("g", {}, svg);
  for (const [a, b] of p.edges) {
    const isNew = freshEdges.has(a + "|" + b);
    arrow(ge, a, b, isNew ? "var(--newlink)" : "var(--link)", isNew ? 3 : 1.8, null, isNew ? "n" + n : "l" + n);
  }
  for (const [a, b] of (p.joint || [])) {
    const isNew = freshEdges.has(a + "|" + b);
    arrow(ge, a, b, isNew ? "var(--newlink)" : "var(--link)", isNew ? 3 : 1.8, "2 2", isNew ? "n" + n : "l" + n);
  }
  for (const [, , b, c] of p.triples) {
    if (inside(b)) { const [r, cc] = parse(b); el("circle", { cx: cx(cc), cy: cy(r), r: S * 0.27, fill: "none", stroke: "var(--out)", "stroke-width": Math.max(2, S * 0.09) }, svg); }
    if (inside(c)) { const [r, cc] = parse(c); el("circle", { cx: cx(cc), cy: cy(r), r: S * 0.11, fill: "var(--exit)" }, svg); }
  }
  /* each triple's exit link id_out -> id_exit - only triples on the drawn chain (candidate_schema) */
  const gx = el("g", {}, svg);
  for (const [, , b, c] of p.triples) if (inside(b) && inside(c)) arrow(gx, b, c, "var(--out)", 1.6, "5 2", "o" + n);
  const head = document.createElement("h2");
  head.textContent = p.label || ("Candidate " + fmt(p.candidate));
  sec.insertBefore(head, box);
  if (p.note) {
    const note = document.createElement("p");
    note.className = "stepnote"; note.textContent = p.note;
    sec.insertBefore(note, box);
  }
  return sec;
}

const panels = document.getElementById("panels"), toc = document.getElementById("toc");
DATA.panels.forEach((p, n) => {
  panels.appendChild(panel(p, n));
  const a = document.createElement("a"); a.href = "#cand-" + n; a.textContent = p.label || fmt(p.candidate); toc.appendChild(a);
});
const show = () => { const on = document.getElementById("show-multi").checked; for (const g of multiGroups) g.style.display = on ? "" : "none"; };
document.getElementById("show-multi").addEventListener("change", show); show();
for (const e of document.querySelectorAll(".steps-only")) e.style.display = DATA.steps ? "" : "none";
document.getElementById("meta").textContent = DATA.steps
  ? `candidate (${DATA.steps}) · ${DATA.panels.length} steps of one pass`
  : `${DATA.tile_label || "tile (" + DATA.tile + ")"} · ${DATA.panels.length} free candidate${DATA.panels.length === 1 ? "" : "s"}` +
    (DATA.other.length ? " · no map for: " + DATA.other.map(([k, st]) => `(${k}) ${st}`).join(", ") : "");
</script>
</html>
"""


# --- one candidate, every step of a pass (2026-11-13) ----------------------------------

def record_candidate_steps(mem, candidate):
    """An observe callback for Pyramid.onepass / upward / downward that
    records, after every step, what `candidate` does at that moment
    (candidate_schema) and what the step added: cells newly forced, links of
    its chain that are new. Returns (observe, steps); pass observe to the
    pass, then steps to render_candidate_steps_html."""
    steps = []

    def observe(label):
        schema = candidate_schema(mem, candidate)
        prev = steps[-1] if steps else {"forced": [], "edges": [], "joint": []}
        new_forced = sorted(set(schema["forced"]) - set(prev["forced"]))
        if steps:
            # no new links into cells that were already forced (Ivo,
            # 2026-11-13): an already forced cell keeps the arrows that
            # explained it in the previous step; only newly forced cells get
            # the current step's arrows
            fresh = set(new_forced)
            schema["edges"] = sorted(prev["edges"] + [e for e in schema["edges"] if e[1] in fresh])
            schema["joint"] = sorted(prev["joint"] + [e for e in schema["joint"] if e[1] in fresh])
        old_edges = {tuple(e) for e in prev["edges"] + prev["joint"]}
        new_edges = sorted(e for e in schema["edges"] + schema["joint"] if tuple(e) not in old_edges)
        if not steps:
            note = f"{len(schema['forced'])} cells forced, {len(schema['edges']) + len(schema['joint'])} links in its chain."
        elif new_forced or new_edges:
            note = (f"{len(new_forced)} newly forced: {' '.join('(' + k + ')' for k in new_forced) or '-'}; "
                    f"{len(new_edges)} new link(s) in its chain.")
        else:
            note = "Nothing new for this candidate."
        steps.append(dict(schema, label=f"{len(steps) + 1}. {label}", note=note,
                          new_forced=new_forced if steps else [], new_edges=new_edges if steps else []))

    return observe, steps


def render_candidate_steps_html(mem, path, title, candidate, steps, note=""):
    """One page, one map per recorded step (record_candidate_steps) of one
    candidate - what it forces after each step of the pass, with what the
    step added highlighted. mem: after the pass (states and tile layout)."""
    base = map_to_schema(mem)
    data = {
        "map": {k: base[k] for k in ("rows", "cols", "tile_size", "origin", "states", "candidates", "suspended")},
        "tiles": {k: v["realm"] for k, v in base["tiles"].items()},
        "tile": "",
        "other": [],
        "steps": _key(candidate),
        "panels": steps,
    }
    page = _CANDIDATE_TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTE__", html.escape(note))
    page = page.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


# --- open seats of a map and how filling them plays out (2026-11-13) --------------------

def render_open_seats_html(path, title, maps, note=""):
    """One page, one section per map: the board as given, with its open
    seats (free cells whose 2x2 block has its three other corners blocked)
    and the corners that make them seats; then the board after filling the
    seats to a fixed point, every filled cell labelled with its round,
    every cell the filling blocked marked, and every 2x2 block left with no
    free or chosen cell outlined. maps: list of dicts with name, rows, cols,
    tile_size, origin, before / after (state grids, rows of 0 free /
    1 chosen / 2 blocked), rounds (per round, a list of {seat, corners,
    blocks}, cells as (row, col)) and holes (upper-left corners)."""
    def key(x):
        return f"{x[0]},{x[1]}"

    data = []
    for mp in maps:
        data.append({
            "name": mp["name"], "rows": mp["rows"], "cols": mp["cols"],
            "tile_size": mp["tile_size"], "origin": mp["origin"],
            "before": mp["before"], "after": mp["after"],
            "rounds": [[{"seat": key(e["seat"]), "corners": [key(c) for c in e["corners"]],
                         "blocks": [key(y) for y in e.get("blocks", [])]} for e in rnd]
                       for rnd in mp["rounds"]],
            "holes": [key(h) for h in mp["holes"]],
        })
    page = _SEATS_TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTE__", html.escape(note))
    page = page.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


_SEATS_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: per map, "as given" and "after filling" side by side, the rounds list below; one column on narrow screens. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1b2230; --muted: #5d6676; --rule: #d5dae2;
  --grid: #e6e9ee; --tile: #2a3346;
  --chosen: #1b2230; --blocked: #c9ced8; --seat: #f2b544; --corner: #c9580f;
  --filled: #2563c9; --newblock: #9db8e8; --hole: #c2352b; --onfill: #ffffff;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
    --grid: #252c37; --tile: #c3cad6;
    --chosen: #e3e7ee; --blocked: #3b4350; --seat: #b07a12; --corner: #f08a45;
    --filled: #6ea0ff; --newblock: #2c4673; --hole: #ff7a6e; --onfill: #12161d; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
  --grid: #252c37; --tile: #c3cad6;
  --chosen: #e3e7ee; --blocked: #3b4350; --seat: #b07a12; --corner: #f08a45;
  --filled: #6ea0ff; --newblock: #2c4673; --hole: #ff7a6e; --onfill: #12161d; color-scheme: dark;
}
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1400px; margin: 0 auto; padding-block: 24px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-family: var(--font-head); font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }
h2 { font-family: var(--font-head); font-size: 1.15rem; margin: 28px 0 4px; }
h3 { font: 600 .72rem var(--font-body); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 6px; }
.lede { color: var(--muted); max-width: 80ch; margin: 0 0 14px; }
.summary { margin: 0 0 12px; max-width: 90ch; }
.summary strong { color: var(--hole); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: .82rem; margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 16px; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 14px; }
.svgbox { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow: auto; }
svg { display: block; }
svg text { font-family: var(--font-mono); }
ol.rounds { font-family: var(--font-mono); font-size: .8rem; margin: 10px 0 0; padding-left: 2.2em; }
ol.rounds li { margin: 2px 0; }
ol.rounds .hit { color: var(--hole); font-weight: 600; }
</style>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__NOTE__</p>
  <div class="legend">
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--chosen)"/></svg>chosen</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--blocked)"/></svg>blocked</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--seat)"/></svg>open seat</span>
    <span><svg viewBox="0 0 26 16"><rect x="7" y="2" width="12" height="12" fill="none" stroke="var(--corner)" stroke-width="2"/></svg>blocked corner making it a seat</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--filled)"/></svg>filled seat (number: round)</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--newblock)"/></svg>blocked by the filling</span>
    <span><svg viewBox="0 0 26 16"><rect x="4" y="1" width="18" height="14" fill="none" stroke="var(--hole)" stroke-width="3"/></svg>2x2 block left uncoverable</span>
  </div>
  <div id="maps"></div>
</div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const parse = k => k.split(",").map(Number);
const fmt = k => "(" + k + ")";
const STATE_FILL = ["none", "var(--chosen)", "var(--blocked)"];

function board(mp, which) {
  const S = Math.max(14, Math.min(26, Math.floor(560 / mp.cols))), O = 22;
  const W = O + mp.cols * S + 4, H = O + mp.rows * S + 4;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": mp.name + (which === "before" ? ": open seats as given" : ": after filling the seats") });
  for (let c = 0; c < mp.cols; c++) { const t = el("text", { x: O + c * S + S / 2, y: 14, "text-anchor": "middle", "font-size": 9, fill: "var(--muted)" }, svg); t.textContent = c; }
  for (let r = 0; r < mp.rows; r++) { const t = el("text", { x: O - 4, y: O + r * S + S / 2 + 3, "text-anchor": "end", "font-size": 9, fill: "var(--muted)" }, svg); t.textContent = r; }
  const round = {}, newly = new Set();
  mp.rounds.forEach((rnd, i) => rnd.forEach(e => { round[e.seat] = i + 1; e.blocks.forEach(y => newly.add(y)); }));
  const first = mp.rounds[0] || [];
  const seatSet = new Set(first.map(e => e.seat)), corners = new Set(first.flatMap(e => e.corners));
  const grid = which === "before" ? mp.before : mp.after;
  for (let r = 0; r < mp.rows; r++) for (let c = 0; c < mp.cols; c++) {
    const k = r + "," + c, st = grid[r][c];
    let fill = STATE_FILL[st];
    if (which === "before" && seatSet.has(k)) fill = "var(--seat)";
    if (which === "after" && round[k]) fill = "var(--filled)";
    else if (which === "after" && newly.has(k)) fill = "var(--newblock)";
    el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill: fill, stroke: "var(--grid)", "stroke-width": 1 }, svg);
    if (which === "before" && corners.has(k)) el("rect", { x: O + c * S + 2.5, y: O + r * S + 2.5, width: S - 5, height: S - 5, fill: "none", stroke: "var(--corner)", "stroke-width": 2 }, svg);
    if (which === "after" && round[k]) { const t = el("text", { x: O + c * S + S / 2, y: O + r * S + S / 2 + 3.5, "text-anchor": "middle", "font-size": Math.min(11, S * 0.5), "font-weight": 600, fill: "var(--onfill)" }, svg); t.textContent = round[k]; }
  }
  const ts = mp.tile_size, org = mp.origin;
  for (let r = org; r <= mp.rows; r += ts) el("line", { x1: O, y1: O + r * S, x2: O + mp.cols * S, y2: O + r * S, stroke: "var(--tile)", "stroke-width": 1.2, opacity: .55 }, svg);
  for (let c = org; c <= mp.cols; c += ts) el("line", { x1: O + c * S, y1: O, x2: O + c * S, y2: O + mp.rows * S, stroke: "var(--tile)", "stroke-width": 1.2, opacity: .55 }, svg);
  if (which === "after") for (const h of mp.holes) { const [r, c] = parse(h); el("rect", { x: O + c * S - 1, y: O + r * S - 1, width: 2 * S + 2, height: 2 * S + 2, fill: "none", stroke: "var(--hole)", "stroke-width": 3.5 }, svg); }
  return svg;
}

const root = document.getElementById("maps");
for (const mp of DATA) {
  const h = document.createElement("h2"); h.textContent = mp.name; root.appendChild(h);
  const filled = mp.rounds.reduce((n, rnd) => n + rnd.length, 0);
  const many = mp.holes.length > 1;
  const p = document.createElement("p"); p.className = "summary";
  p.innerHTML = `${(mp.rounds[0] || []).length} open seat(s) as given; filling them chooses ${filled} cell(s) in ${mp.rounds.length} round(s).` +
    (mp.holes.length ? ` <strong>Afterwards the 2x2 block${many ? "s" : ""} at ${mp.holes.map(fmt).join(", ")} ha${many ? "ve" : "s"} no free or chosen cell - nothing can cover ${many ? "them" : "it"}; this board has no valid completion.</strong>` : " No 2x2 block is left uncoverable.");
  root.appendChild(p);
  const pair = document.createElement("div"); pair.className = "pair"; root.appendChild(pair);
  for (const [which, label] of [["before", "as given - open seats"], ["after", "after filling the seats"]]) {
    const box = document.createElement("div"); box.className = "svgbox";
    const h3 = document.createElement("h3"); h3.textContent = label; box.appendChild(h3);
    box.appendChild(board(mp, which)); pair.appendChild(box);
  }
  const holeCells = new Set(mp.holes.flatMap(h => { const [r, c] = parse(h); return [[r, c], [r, c + 1], [r + 1, c], [r + 1, c + 1]].map(x => x.join(",")); }));
  const ol = document.createElement("ol"); ol.className = "rounds"; ol.setAttribute("aria-label", "Rounds of filling");
  mp.rounds.forEach(rnd => {
    const li = document.createElement("li");
    li.innerHTML = rnd.map(e =>
      `${fmt(e.seat)} - corners ${e.corners.map(fmt).join(" ")} blocked; blocks ` +
      (e.blocks.length ? e.blocks.map(y => holeCells.has(y) ? `<span class="hit">${fmt(y)}</span>` : fmt(y)).join(" ") : "-")
    ).join("<br>");
    ol.appendChild(li);
  });
  root.appendChild(ol);
}
</script>
</body>
</html>
"""


# --- where each candidate's chain enters each tile (2026-11-14) ------------------------

def candidate_entries(mem):
    """{candidate: set of entry cells} after a pass - the cells where the
    candidate's chain enters a base tile (2026-11-14, Ivo: "the result we
    are interested in [is] the entries for each tile from (possibly
    far-away) candidates"): a cell y of tile T is an entry of c if c
    forces y (Base.claimed: single links and n-ary facts) and y is
    - forced through a single link from a cell of c's chain outside T, or
    - one of c's n-ary collapse targets (mem.collapsed - the fact the
      n-ary step gave to T), or
    - c itself (its own tile).
    Cells outside every tile are left out. Free, not suspended candidates
    only."""
    m = mem.map
    base = mem.pyramid[0]
    size, origin = mem.tile_size, mem.origin

    def tile_of(x):
        if x[0] < origin or x[1] < origin:
            return None
        t = ((x[0] - origin) // size, (x[1] - origin) // size)
        return t if t[0] < base.shape[0] and t[1] < base.shape[1] else None

    forcers = Base._single_forcers(m)
    entries = {}
    for c in np.ndindex(m.shape):
        if m[c].state != StateEnum.free or not Pyramid.is_candidate(c):
            continue
        chain = Base.claimed(mem, c, forcers) | {c}
        cells = {c} | {y for (cc, y) in mem.collapsed if cc == c}
        for x in chain:
            for y in forcers.get(x, ()):
                if y in chain and tile_of(x) != tile_of(y):
                    cells.add(y)
        entries[c] = {y for y in cells if tile_of(y) is not None}
    return entries


def render_entries_html(mem, path, title, note=""):
    """One map (mem after a pass): every placement tile gets a colour of
    its own, every candidate in it a shade of that colour, and each
    candidate's entries (candidate_entries) are dots of its shade - several
    candidates entering the same cell get their dots shifted apart. Nothing
    else: the states, the tile grid, the placement tiles outlined in their
    colour, and a legend naming the candidates."""
    m = mem.map
    rows, cols = m.shape
    size, origin = mem.tile_size, mem.origin
    entries = candidate_entries(mem)
    ptiles = sorted({((c[0] - cores.PLACEMENT_ORIGIN) // cores.PLACEMENT_TILE_SIZE,
                      (c[1] - cores.PLACEMENT_ORIGIN) // cores.PLACEMENT_TILE_SIZE) for c in entries})
    data = {
        "rows": rows, "cols": cols, "tile_size": size, "origin": origin,
        "states": [[_STATE[m[i, j].state] for j in range(cols)] for i in range(rows)],
        "ptiles": [{"tile": list(t),
                    "realm": [[cores.PLACEMENT_ORIGIN + t[0] * cores.PLACEMENT_TILE_SIZE,
                               cores.PLACEMENT_ORIGIN + (t[0] + 1) * cores.PLACEMENT_TILE_SIZE],
                              [cores.PLACEMENT_ORIGIN + t[1] * cores.PLACEMENT_TILE_SIZE,
                               cores.PLACEMENT_ORIGIN + (t[1] + 1) * cores.PLACEMENT_TILE_SIZE]],
                    "candidates": [[_key(c), sorted(_key(y) for y in entries[c])]
                                   for c in sorted(entries)
                                   if ((c[0] - cores.PLACEMENT_ORIGIN) // cores.PLACEMENT_TILE_SIZE,
                                       (c[1] - cores.PLACEMENT_ORIGIN) // cores.PLACEMENT_TILE_SIZE) == t]}
                   for t in ptiles],
    }
    page = _ENTRIES_TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTE__", html.escape(note))
    page = page.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


_ENTRIES_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: the map, the legend (one row per placement tile) beside it; one column on narrow screens. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1b2230; --muted: #5d6676; --rule: #d5dae2;
  --grid: #e6e9ee; --tile: #2a3346; --chosen: #1b2230; --blocked: #c9ced8; --ring: #ffffff;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
    --grid: #252c37; --tile: #c3cad6; --chosen: #e3e7ee; --blocked: #3b4350; --ring: #1a2029; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
  --grid: #252c37; --tile: #c3cad6; --chosen: #e3e7ee; --blocked: #3b4350; --ring: #1a2029; color-scheme: dark;
}
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1400px; margin: 0 auto; padding-block: 24px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-family: var(--font-head); font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }
.lede { color: var(--muted); max-width: 80ch; margin: 0 0 14px; }
.layout { display: grid; grid-template-columns: auto minmax(220px, 380px); justify-content: start; gap: 18px; align-items: start; }
@media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }
.svgbox { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow: auto; justify-self: start; max-width: 100%; }
svg { display: block; }
svg text { font-family: var(--font-mono); }
.legend { display: grid; gap: 10px; font-size: .82rem; }
.legend .row h2 { font: 600 .72rem var(--font-body); text-transform: uppercase; letter-spacing: .08em; margin: 0 0 4px; }
.legend .chips { display: flex; flex-wrap: wrap; gap: 4px 10px; }
.legend .chip { display: inline-flex; align-items: center; gap: 5px; font-family: var(--font-mono); cursor: default; border-radius: 4px; padding: 0 3px; }
.legend .chip .dot { width: 11px; height: 11px; border-radius: 50%; }
.legend .chip .n { color: var(--muted); }
.legend .chip.on { outline: 2px solid var(--fg); }
.dim { opacity: .12; }
</style>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__NOTE__</p>
  <div class="layout">
    <div class="svgbox" id="box"></div>
    <div class="legend" id="legend" aria-label="Placement tiles and their candidates"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const parse = k => k.split(",").map(Number);
const dark = () => getComputedStyle(document.documentElement).colorScheme === "dark";

/* one hue per placement tile, spread around the wheel; shades of it per candidate */
const nT = DATA.ptiles.length;
const color = {};
DATA.ptiles.forEach((pt, i) => {
  const hue = Math.round((i * 360 / nT + 8) % 360), n = pt.candidates.length;
  pt.hue = hue;
  pt.candidates.forEach(([c], j) => {
    const t = n === 1 ? 0.5 : j / (n - 1);
    const light = dark() ? 42 + 38 * t : 24 + 46 * t;
    color[c] = `hsl(${(hue + 30 * (t - 0.5) + 360) % 360} ${86 - 22 * t}% ${light}%)`;
  });
});

const S = Math.max(16, Math.min(34, Math.floor(820 / DATA.cols))), O = 22;
const W = O + DATA.cols * S + 4, H = O + DATA.rows * S + 4;
const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Entries of every candidate, coloured by placement tile" });
document.getElementById("box").appendChild(svg);
for (let c = 0; c < DATA.cols; c++) { const t = el("text", { x: O + c * S + S / 2, y: 14, "text-anchor": "middle", "font-size": 9, fill: "var(--muted)" }, svg); t.textContent = c; }
for (let r = 0; r < DATA.rows; r++) { const t = el("text", { x: O - 4, y: O + r * S + S / 2 + 3, "text-anchor": "end", "font-size": 9, fill: "var(--muted)" }, svg); t.textContent = r; }
const FILL = ["none", "var(--chosen)", "var(--blocked)"];
for (let r = 0; r < DATA.rows; r++) for (let c = 0; c < DATA.cols; c++)
  el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill: FILL[DATA.states[r][c]], stroke: "var(--grid)", "stroke-width": 1 }, svg);
const ts = DATA.tile_size, org = DATA.origin;
for (let r = org; r <= DATA.rows; r += ts) el("line", { x1: O + Math.max(0, org) * S, y1: O + r * S, x2: O + DATA.cols * S, y2: O + r * S, stroke: "var(--tile)", "stroke-width": 1, opacity: .45 }, svg);
for (let c = org; c <= DATA.cols; c += ts) el("line", { x1: O + c * S, y1: O + Math.max(0, org) * S, x2: O + c * S, y2: O + DATA.rows * S, stroke: "var(--tile)", "stroke-width": 1, opacity: .45 }, svg);
for (const pt of DATA.ptiles) {
  const [[r0, r1], [c0, c1]] = pt.realm;
  el("rect", { x: O + c0 * S + 1.5, y: O + r0 * S + 1.5, width: (c1 - c0) * S - 3, height: (r1 - r0) * S - 3, fill: "none",
    stroke: `hsl(${pt.hue} 78% ${dark() ? 60 : 42}%)`, "stroke-width": 3 }, svg);
}

/* dots: every candidate entering a cell, shifted apart around the cell centre */
const at = {};
for (const pt of DATA.ptiles) for (const [c, cells] of pt.candidates) for (const y of cells) (at[y] = at[y] || []).push(c);
const dots = [];
for (const y in at) {
  const [r, c] = parse(y), cs = at[y], k = cs.length;
  const cx = O + c * S + S / 2, cy = O + r * S + S / 2;
  const rad = k === 1 ? 0 : S * (k <= 4 ? 0.2 : 0.27), dr = Math.max(2.4, S * (k <= 4 ? 0.15 : k <= 8 ? 0.11 : 0.085));
  cs.forEach((cand, i) => {
    const a = -Math.PI / 2 + 2 * Math.PI * i / k;
    const d = el("circle", { cx: cx + rad * Math.cos(a), cy: cy + rad * Math.sin(a), r: dr, fill: color[cand], stroke: "var(--ring)", "stroke-width": 1 }, svg);
    const tt = el("title", {}, d); tt.textContent = `(${cand}) enters (${y})`;
    d.dataset.cand = cand; dots.push(d);
  });
}

/* legend: per placement tile its candidates; hover or tap one to show only its dots */
const legend = document.getElementById("legend");
let pinned = null;
const focus = cand => { for (const d of dots) d.classList.toggle("dim", cand !== null && d.dataset.cand !== cand);
  for (const ch of legend.querySelectorAll(".chip")) ch.classList.toggle("on", ch.dataset.cand === cand); };
for (const pt of DATA.ptiles) {
  const row = document.createElement("div"); row.className = "row";
  const h = document.createElement("h2"); h.textContent = `placement tile (${pt.tile})`; h.style.color = `hsl(${pt.hue} 78% ${dark() ? 62 : 38}%)`;
  row.appendChild(h);
  const chips = document.createElement("div"); chips.className = "chips";
  for (const [c, cells] of pt.candidates) {
    const ch = document.createElement("span"); ch.className = "chip"; ch.dataset.cand = c; ch.tabIndex = 0;
    ch.innerHTML = `<span class="dot" style="background:${color[c]}"></span>(${c}) <span class="n">${cells.length}</span>`;
    ch.addEventListener("mouseenter", () => pinned === null && focus(c));
    ch.addEventListener("mouseleave", () => pinned === null && focus(null));
    ch.addEventListener("click", () => { pinned = pinned === c ? null : c; focus(pinned); });
    ch.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); ch.click(); } });
    chips.appendChild(ch);
  }
  row.appendChild(chips); legend.appendChild(row);
}
</script>
</body>
</html>
"""


# --- blocking and self-blocking, marked on the map (2026-11-16) ------------------------

def render_blocking_html(mem, path, title, note=""):
    """One page (mem after a pass), one map per situation from the base
    tiles' .blocked lists (2026-11-16, Ivo: "find a situation where a path
    from a candidate blocks the path of a candidate from another placement
    tile, and a candidate that is self-blocking. Mark the blocking cells and
    the blocked candidates in the maps"):
    - every blocking pair (c1, c2): the cells each forces (Base.claimed)
      tinted in its own colour, both candidates outlined, and every pair of
      cells that block each other (Base.blocking_cells) marked red and
      joined;
    - every refuted candidate c (mem.refuted, the pass's report - suspended
      in the pass, to be blocked by the CPU's unblocking step): its chain and
      its blocking cell pairs, up to its first contradiction."""
    m = mem.map
    rows, cols = m.shape
    blocked = set().union(*(t.blocked for t in mem.pyramid[0].flat))
    forcers = Base._single_forcers(m)

    def ptile(c):
        return f"{(c[0] - cores.PLACEMENT_ORIGIN) // cores.PLACEMENT_TILE_SIZE},"\
               f"{(c[1] - cores.PLACEMENT_ORIGIN) // cores.PLACEMENT_TILE_SIZE}"

    panels = []
    for c1, c2 in sorted(p for p in blocked if p[0] != p[1]):
        panels.append({
            "kind": "pair", "cands": [_key(c1), _key(c2)], "tiles": [ptile(c1), ptile(c2)],
            "chains": [sorted(_key(x) for x in Base.claimed(mem, c, forcers)) for c in (c1, c2)],
            "pairs": sorted([_key(x), _key(y)] for x, y in Base.blocking_cells(mem, c1, c2)),
        })
    for c, (chain, pairs) in sorted(mem.refuted.items()):
        panels.append({
            "kind": "self", "cands": [_key(c)], "tiles": [ptile(c)],
            "chains": [sorted(_key(x) for x in chain if x != c)],
            "pairs": sorted([_key(x), _key(y)] for x, y in pairs),
        })
    base = map_to_schema(mem)
    data = {
        "map": {k: base[k] for k in ("rows", "cols", "states", "candidates")},
        "tiles": {k: v["realm"] for k, v in base["tiles"].items()},
        "panels": panels,
    }
    page = _BLOCKING_TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTE__", html.escape(note))
    page = page.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


_BLOCKING_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: a grid of map cards, one per situation; one column on narrow screens. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1b2230; --muted: #5d6676; --rule: #d5dae2;
  --grid: #e6e9ee; --tile: #2a3346; --place: #fff1d6; --chosen: #1b2230; --blocked: #c9ced8;
  --a: #2563c9; --b: #d9730d; --self: #8a3fc4; --hit: #d0202e;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
    --grid: #252c37; --tile: #c3cad6; --place: #3a2f17; --chosen: #e3e7ee; --blocked: #3b4350;
    --a: #6ea0ff; --b: #f0a050; --self: #b98cf0; --hit: #ff5a64; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --panel: #1a2029; --fg: #e3e7ee; --muted: #9aa3b2; --rule: #2e3644;
  --grid: #252c37; --tile: #c3cad6; --place: #3a2f17; --chosen: #e3e7ee; --blocked: #3b4350;
  --a: #6ea0ff; --b: #f0a050; --self: #b98cf0; --hit: #ff5a64; color-scheme: dark;
}
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1500px; margin: 0 auto; padding-block: 24px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-family: var(--font-head); font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }
h2 { font-family: var(--font-head); font-size: 1.02rem; margin: 0 0 2px; }
.lede { color: var(--muted); max-width: 85ch; margin: 0 0 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: .82rem; margin: 0 0 16px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 430px), 1fr)); gap: 14px; }
.card { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 10px; overflow: auto; }
.sub { color: var(--muted); font-size: .82rem; margin: 0 0 6px; font-family: var(--font-mono); }
svg { display: block; }
svg text { font-family: var(--font-mono); }
</style>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__NOTE__</p>
  <div class="legend">
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--chosen)"/></svg>chosen</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--blocked)"/></svg>blocked</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--place)" stroke="var(--rule)"/></svg>eligible cells</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--a)" fill-opacity=".35"/></svg>forced by the first candidate</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--b)" fill-opacity=".35"/></svg>forced by the second candidate</span>
    <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="var(--self)" fill-opacity=".35"/></svg>forced by the self-blocking candidate</span>
    <span><svg viewBox="0 0 26 16"><rect x="5" y="0" width="16" height="16" fill="none" stroke="var(--a)" stroke-width="3"/></svg>blocked candidate</span>
    <span><svg viewBox="0 0 26 16"><circle cx="5" cy="12" r="3.5" fill="none" stroke="var(--hit)" stroke-width="2"/><circle cx="21" cy="4" r="3.5" fill="none" stroke="var(--hit)" stroke-width="2"/><line x1="7.5" y1="10" x2="18.5" y2="6" stroke="var(--hit)" stroke-width="2"/></svg>cells that block each other</span>
  </div>
  <div class="grid" id="grid"></div>
</div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const parse = k => k.split(",").map(Number);
const M = DATA.map, isCand = new Set(M.candidates);
const S = Math.max(12, Math.min(24, Math.floor(380 / M.cols))), O = 20;
const FILL = ["none", "var(--chosen)", "var(--blocked)"];
const cx = c => O + c * S + S / 2, cy = r => O + r * S + S / 2;

function map(p) {
  const W = O + M.cols * S + 4, H = O + M.rows * S + 4;
  const what = p.kind === "pair" ? `(${p.cands[0]}) and (${p.cands[1]}) blocking each other` : `(${p.cands[0]}) blocking itself`;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": what });
  for (let c = 0; c < M.cols; c++) { const t = el("text", { x: cx(c), y: 13, "text-anchor": "middle", "font-size": 8, fill: "var(--muted)" }, svg); t.textContent = c; }
  for (let r = 0; r < M.rows; r++) { const t = el("text", { x: O - 3, y: cy(r) + 3, "text-anchor": "end", "font-size": 8, fill: "var(--muted)" }, svg); t.textContent = r; }
  for (const k in DATA.tiles) { const [[r0, r1], [c0, c1]] = DATA.tiles[k];
    if (isCand.has(r0 + "," + c0)) el("rect", { x: O + c0 * S, y: O + r0 * S, width: (c1 - c0) * S, height: (r1 - r0) * S, fill: "var(--place)" }, svg); }
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    const st = (p.states || M.states)[r][c];
    el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill: st ? FILL[st] : "none", stroke: "var(--grid)", "stroke-width": 1 }, svg);
  }
  const colours = p.kind === "pair" ? ["var(--a)", "var(--b)"] : ["var(--self)"];
  p.chains.forEach((chain, i) => { for (const k of chain) { const [r, c] = parse(k);
    el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill: colours[i], "fill-opacity": .35 }, svg); } });
  for (const k in DATA.tiles) { const [[r0, r1], [c0, c1]] = DATA.tiles[k];
    el("rect", { x: O + c0 * S, y: O + r0 * S, width: (c1 - c0) * S, height: (r1 - r0) * S, fill: "none", stroke: "var(--tile)", "stroke-width": 1.3 }, svg); }
  p.cands.forEach((k, i) => { const [r, c] = parse(k);
    el("rect", { x: O + c * S + 1.5, y: O + r * S + 1.5, width: S - 3, height: S - 3, fill: "none", stroke: colours[i], "stroke-width": 3.2 }, svg); });
  const hit = new Set(p.pairs.flat());
  for (const [a, b] of p.pairs) { const [r1, c1] = parse(a), [r2, c2] = parse(b);
    el("line", { x1: cx(c1), y1: cy(r1), x2: cx(c2), y2: cy(r2), stroke: "var(--hit)", "stroke-width": 2.4 }, svg); }
  for (const k of hit) { const [r, c] = parse(k);
    el("circle", { cx: cx(c), cy: cy(r), r: S * 0.3, fill: "none", stroke: "var(--hit)", "stroke-width": 2.2 }, svg); }
  return svg;
}

const grid = document.getElementById("grid");
for (const p of DATA.panels) {
  const card = document.createElement("div"); card.className = "card";
  const h = document.createElement("h2");
  h.textContent = p.kind === "pair" ? `(${p.cands[0]}) and (${p.cands[1]}) block each other` : `(${p.cands[0]}) blocks itself`;
  const sub = document.createElement("p"); sub.className = "sub";
  sub.textContent = (p.kind === "pair" ? `placement tiles (${p.tiles[0]}) and (${p.tiles[1]})` : `placement tile (${p.tiles[0]}) - refuted: suspended in the pass, blocked by the CPU's unblocking step`) +
    ` · ${p.pairs.length} blocking cell pair${p.pairs.length === 1 ? "" : "s"}`;
  card.append(h, sub, map(p)); grid.appendChild(card);
}
if (!DATA.panels.length) { const e = document.createElement("p"); e.className = "lede"; e.textContent = "No blocking pair and no self-blocking candidate."; grid.appendChild(e); }
</script>
</body>
</html>
"""
