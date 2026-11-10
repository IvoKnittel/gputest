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
- mem.pyramid[0], if built: each base tile's .links triples, listed per tile
  and drawn on the map (fat rings at id_out, filled dots at id_in, small
  dots at id_exit; clicking a tile card highlights only that tile's);
- entries (optional): each base tile's simultaneous entries - a dict
  {(ci, cj): set of (candidate, cell)} or an array laid out like
  mem.pyramid[0] - listed per tile.

The page is self-contained (data embedded as JSON, drawing in inline JS),
light and dark theme, nothing loaded at runtime except Google Fonts.
"""
import html
import json

import numpy as np

from cores import Pyramid, DUMMY
from map_of_squares import StateEnum

_STATE = {StateEnum.free: 0, StateEnum.chosen: 1, StateEnum.blocked: 2}


def _key(id_):
    return id_ if id_ == DUMMY else f"{id_[0]},{id_[1]}"


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
                "links": sorted([_key(a), _key(b), _key(c)] for a, b, c in tile.links),
                "entries": sorted([_key(a), _key(b)] for a, b in tile_entries),
            }
    return {
        "rows": rows, "cols": cols,
        "tile_size": mem.tile_size, "origin": mem.origin,
        "states": states,
        "candidates": candidates,
        "singles": sorted(singles),
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
.layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 520px); gap: 24px; align-items: start; }
@media (max-width: 1100px) { .layout { grid-template-columns: minmax(0, 1fr); } }
.figure { display: grid; gap: 10px; position: sticky; top: 12px; }
@media (max-width: 1100px) { .figure { position: static; } }
.svgbox { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow: auto; max-height: 86vh; }
svg { display: block; }
svg text { font-family: var(--font-mono); }
.controls { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .88rem; }
.controls label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: .82rem; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 8px; }
.card { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; min-width: 0; cursor: pointer; }
.card:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }
.card.place { background: linear-gradient(var(--place), var(--place)) top / 100% 4px no-repeat, var(--panel); }
.card.active { border-color: var(--fg); }
.card h3 { font: 600 .88rem var(--font-head); margin: 0 0 4px; display: flex; justify-content: space-between; gap: 8px; }
.card h3 small { font: 400 .72rem var(--font-mono); color: var(--muted); }
.label { font: 600 .66rem var(--font-body); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-top: 4px; }
.item { font-family: var(--font-mono); font-size: .76rem; overflow-wrap: anywhere; }
.empty { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); }
.cnt { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
</style>

<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="lede">__NOTE__</p>
  <p class="stats" id="stats"></p>
  <div class="layout">
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
        <span><svg viewBox="0 0 26 16"><rect x="6" y="1" width="14" height="14" fill="none" stroke="var(--cand)" stroke-width="2" stroke-dasharray="3 2"/></svg>free candidate</span>
        <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--link)" stroke-width="2"/></svg>single link</span>
        <span><svg viewBox="0 0 26 16"><line x1="2" y1="8" x2="24" y2="8" stroke="var(--pair)" stroke-width="1.5" stroke-dasharray="3 2"/></svg>pair condition</span>
        <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="5" fill="var(--in)"/></svg>id_in</span>
        <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="5" fill="none" stroke="var(--out)" stroke-width="3"/></svg>id_out</span>
        <span><svg viewBox="0 0 26 16"><circle cx="13" cy="8" r="3" fill="var(--exit)"/></svg>id_exit</span>
      </div>
    </div>
    <div>
      <h2>Base tiles</h2>
      <p class="cnt">Click a tile to show only its triples on the map; click again to show all.</p>
      <div class="tiles" id="tiles"></div>
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
  el("rect", { x: O + c * S, y: O + r * S, width: S, height: S, fill: st === 1 ? "var(--chosen)" : st === 2 ? "var(--blocked)" : "none", stroke: "var(--grid)", "stroke-width": 1 });
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

/* tile cards */
const fmt = k => k === "dummy" ? "dummy" : "(" + k + ")";
const box = document.getElementById("tiles");
let active = null;
const keys = Object.keys(DATA.tiles).sort((a, b) => { const [a1, a2] = parse(a), [b1, b2] = parse(b); return a1 - b1 || a2 - b2; });
for (const k of keys) {
  const t = DATA.tiles[k], [[r0, r1], [c0, c1]] = t.realm;
  const card = document.createElement("div");
  card.className = "card" + (isCand.has(r0 + "," + c0) ? " place" : "");
  card.tabIndex = 0; card.dataset.k = k;
  const links = t.links.length ? t.links.map(([a, b, c]) => `<div class="item">(${fmt(a)}, ${fmt(b)}, ${fmt(c)})</div>`).join("") : '<div class="empty">none</div>';
  const ents = t.entries.length ? t.entries.map(([a, b]) => `<div class="item">(${fmt(a)}, ${fmt(b)})</div>`).join("") : '<div class="empty">none</div>';
  const nEntCand = new Set(t.entries.map(e => e[0])).size;
  card.innerHTML = `<h3>Tile (${k})<small>rows ${r0}–${r1 - 1}, cols ${c0}–${c1 - 1}</small></h3>` +
    `<div class="label">.links (${t.links.length})</div>${links}` +
    `<div class="label">simultaneous entries (${t.entries.length}, from ${nEntCand} candidate${nEntCand === 1 ? "" : "s"})</div>${ents}`;
  const pick = () => {
    active = active === k ? null : k;
    for (const c of box.children) c.classList.toggle("active", c.dataset.k === active);
    drawTileLinks(active);
  };
  card.addEventListener("click", pick);
  card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  box.appendChild(card);
}

const nTripleCells = Object.keys(DATA.triple_count).length;
const multi = keys.filter(k => new Set(DATA.tiles[k].entries.map(e => e[0])).size >= 2).length;
document.getElementById("stats").textContent =
  `${R}×${C} cells · ${keys.length} base tiles of ${ts}×${ts} from origin ${org} · ` +
  `${DATA.singles.length} single links · ${DATA.pairs.length} pair conditions · ` +
  `${nTripleCells} cells with triple conditions · ${multi} tiles with entries from 2+ candidates`;
drawTileLinks(null); vis();
</script>
</html>
"""
