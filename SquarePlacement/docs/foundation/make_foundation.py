"""Builds foundation.html: the figures' data are computed here (examples.py,
engine.py - vacuum conditions and compute_conditions only) and injected into
foundation_template.html. Run: python make_foundation.py"""
import json
import os

from engine import initial_map, choose, compute_conditions, states, links, diagonals, CHOSEN, BLOCKED
from examples import (ex_conditions, ex_place, ex_seat, ex_path, ex_pair, ex_refute, defect_count,
                      PINWHEEL, PINWHEEL_BLOCK, terms)

HERE = os.path.dirname(os.path.abspath(__file__))


def key(p):
    return f"{p[0]},{p[1]}"


def fig(st, rows, cols, **kw):
    d = dict(st=st, rows=rows, cols=cols, cells={}, labels={}, arrows=[], rings=[], pixels=[], blocks=[])
    for k, v in kw.items():
        d[k] = v
    return d


def mark(d, cells, cls):
    for p in cells:
        d["cells"].setdefault(key(p), []).append(cls)


def build():
    F, facts = {}, {}

    # 1 conditions
    e = ex_conditions()
    y, t = e["y"], e["term"]
    d = fig(states(e["m"]), [0, 6], [0, 6])
    mark(d, [y], "focus")
    d["labels"][key(y)] = "y"
    blocker_of = {k: PINWHEEL[k] for k in PINWHEEL_BLOCK if k != y}
    mark(d, blocker_of.keys(), "corner")
    mark(d, t, "member")
    for k, b in blocker_of.items():
        d["arrows"].append([b, k, "blocks"])
    d["blocks"].append([2, 2, "seat"])
    d["pixels"].append([3, 3, "focus"])
    F["cond"] = d
    facts.update(n_terms=e["n_terms"], term=t, defects=defect_count())

    # 2 + 3 placing, blocked diagonals
    e = ex_place()
    p = e["p"]
    d = fig(states(e["m"]), [1, 7], [1, 7])
    mark(d, [p], "focus")
    d["labels"][key(p)] = "p"
    d["pixels"] += [[p[0] + a, p[1] + b, "covered"] for a in (0, 1) for b in (0, 1)]
    for q in e["diag"]:
        d["arrows"].append([p, q, "blocks"])
    F["place"] = d
    facts.update(with_p=e["with_p"], with_diag=e["with_diag"], own=e["own"], pairs_after=e["pairs_after"])

    # sample terms for 2 and 3: a term containing p (before/after) and one containing a diagonal
    m = initial_map((9, 9))
    q = (3, 3)
    w = (2, 5)          # a cell with a term containing p=(4,4)
    tw = next(sorted(x for x, _ in tt) for tt in sorted(m[w].forced_by, key=lambda tt: sorted(x for x, _ in tt))
              if any(x == p for x, _ in tt) and not any(x in e["diag"] for x, _ in tt))
    v = (1, 2)          # a cell with a term containing q=(3,3)
    tv = next(sorted(x for x, _ in tt) for tt in sorted(m[v].forced_by, key=lambda tt: sorted(x for x, _ in tt))
              if any(x == q for x, _ in tt))
    choose(m, p)
    compute_conditions(m)
    tw_after = [x for x in tw if x != p]
    assert tw_after in terms(m, w)
    assert not any(any(x == q for x, _ in tt) for c in [(i, j) for i in range(9) for j in range(9)]
                   for tt in m[c].forced_by)
    facts.update(w=w, tw=tw, tw_after=tw_after, v=v, tv=tv, q=q)
    d = fig(states(m), [0, 7], [0, 7])
    mark(d, [q], "focus")
    d["labels"][key(q)] = "q"
    mark(d, [x for x in tv if x != q], "member")
    d["labels"][key(v)] = "v"
    mark(d, [v], "head")
    F["blocked"] = d

    # 4 the hole (hypothetical) and the legal outcome
    st = [[0] * 7 for _ in range(7)]
    for b in PINWHEEL.values():
        st[b[0]][b[1]] = 1
    for k in PINWHEEL_BLOCK:
        st[k[0]][k[1]] = 2
    mh = initial_map((7, 7))
    for b in PINWHEEL.values():
        for dd in diagonals(mh, b):
            if st[dd[0]][dd[1]] == 0:
                st[dd[0]][dd[1]] = 2
    d = fig(st, [0, 6], [0, 6])
    for k, b in PINWHEEL.items():
        d["arrows"].append([b, k, "blocks"])
    d["blocks"].append([2, 2, "dead"])
    d["pixels"].append([3, 3, "dead"])
    F["hole"] = d
    m = initial_map((7, 7))
    for b in [(1, 2), (3, 1), (4, 3)]:
        choose(m, b)
    from engine import settle
    placed = settle(m)
    assert (3, 3) in placed and m[2, 4].state == BLOCKED
    d = fig(states(m), [0, 6], [0, 6])
    mark(d, [(3, 3)], "focus")
    d["labels"]["3,3"] = "y"
    mark(d, [(2, 4)], "ghost")
    d["arrows"].append([(3, 3), (2, 4), "blocks"])
    d["pixels"].append([3, 3, "covered"])
    F["hole_legal"] = d

    # 5 seat frames
    frames = ex_seat()
    F["seat"] = []
    for k, fr in enumerate(frames):
        d = fig(fr["st"], [0, 6], [0, 6])
        mark(d, [(3, 3)], "focus")
        d["labels"]["3,3"] = "y"
        mark(d, fr["term"], "member")
        d["blocks"].append([2, 2, "seat" if k == 3 else "plain"])
        if k == 2:
            for a, yy in fr["links"]:
                if yy == (3, 3):
                    d["arrows"].append([a, yy, "link"])
        F["seat"].append(d)
    facts["seat_links"] = [a for a, yy in frames[2]["links"] if yy == (3, 3)]

    # 6 path
    e = ex_path()
    z = e["z"]
    d = fig(states(e["m"]), [1, 9], [2, 10])
    mark(d, [x for x in e["chain"] if x != z], "alias")
    mark(d, [z], "cand")
    d["labels"][key(z)] = "Z"
    for a, yy in e["links"]:
        d["arrows"].append([a, yy, "link"])
    F["path"] = d
    seat = e["seats"][(z, (3, 5))]
    d = fig(states(e["m"]), [1, 7], [2, 9])
    mark(d, [z], "cand")
    d["labels"][key(z)] = "Z"
    mark(d, [(3, 5)], "focus")
    for k, b in seat.items():
        if b != z:
            d["arrows"].append([b, k, "blocks"])
        else:
            d["arrows"].append([b, k, "blocks-z"])
        mark(d, [k], "corner")
    d["blocks"].append([min(k[0] for k in list(seat) + [(3, 5)]), min(k[1] for k in list(seat) + [(3, 5)]), "seat"])
    d["arrows"].append([z, (3, 5), "link"])
    F["path_seat"] = d
    depth = {z: 0}
    order = [z]
    fwd = {}
    for a, yy in e["links"]:
        fwd.setdefault(a, []).append(yy)
    for x in order:
        for yy in sorted(fwd.get(x, [])):
            if yy not in depth:
                depth[yy] = depth[x] + 1
                order.append(yy)
    F["path_graph"] = dict(nodes=[dict(id=key(x), label=str(x), layer=depth[x], cand=(x == z)) for x in order],
                           edges=[[key(a), key(yy)] for a, yy in e["links"]])
    facts.update(path_z=z, path_chain=e["chain"], path_links=e["links"], path_seat=seat)

    # 7 candidate placement
    before_links, after_links = links(e["m"]), links(e["after"])
    d = fig(states(e["after"]), [1, 9], [2, 10])
    mark(d, [x for x in e["chain"] if x != z], "alias")
    mark(d, [z], "cand")
    d["labels"][key(z)] = "Z"
    newly = [(i, j) for i in range(11) for j in range(11)
             if e["m"][i, j].state != BLOCKED and e["after"][i, j].state == BLOCKED]
    mark(d, newly, "newblocked")
    F["cand_after"] = d
    facts.update(n_links_before=len(before_links), n_links_after=len(after_links),
                 n_newly_blocked=len(newly), cand_forced=e["forced"])

    # 8 blocked pair
    e = ex_pair()
    z1, z2 = e["z1"], e["z2"]
    d = fig(states(e["m"]), [2, 9], [0, 7])
    mark(d, [x for x in e["c1"] if x != z1], "alias")
    mark(d, [x for x in e["c2"] if x != z2], "alias2")
    mark(d, [z1], "cand")
    mark(d, [z2], "cand2")
    d["labels"][key(z1)] = "Z₁"
    d["labels"][key(z2)] = "Z₂"
    for a, yy in e["links"]:
        d["arrows"].append([a, yy, "link"])
    d["rings"] = [[x, yy] for x, yy in e["pairs"]]
    F["pair"] = d
    blk = e["seats"][(6, 3)][1]
    d = fig(states(e["m"]), [2, 9], [0, 7])
    mark(d, [z1], "cand")
    mark(d, [z2], "cand2")
    d["labels"][key(z1)] = "Z₁"
    d["labels"][key(z2)] = "Z₂"
    for k, b in blk.items():
        d["arrows"].append([b, k, "blocks-z" if b == z1 else "blocks"])
    d["arrows"].append([z2, (6, 3), "blocks-z2"])
    d["blocks"].append([5, 3, "dead"])
    d["pixels"].append([6, 4, "dead"])
    F["pair_defect"] = d
    facts.update(pair_z1=z1, pair_z2=z2, pair_c1=e["c1"], pair_c2=e["c2"], pair_pairs=e["pairs"],
                 pair_seat_63=e["seats"][(6, 3)], pair_seat_53=e["seats"][(5, 3)])

    # 9 refutation
    e = ex_refute()
    z = e["z"]
    d = fig(states(e["m"]), [1, 8], [1, 8])
    mark(d, e["first"], "alias")
    mark(d, [x for x in e["chain"] if x not in e["first"] and x != z], "faded")
    mark(d, [z], "cand")
    d["labels"][key(z)] = "Z"
    for a, yy in e["links"]:
        d["arrows"].append([a, yy, "link" if a == z else "link-faded"])
    d["rings"] = [list(pp) for pp in e["first_pairs"]]
    F["refute"] = d
    seat = e["seats"][(4, 5)]
    d = fig(states(e["m"]), [1, 8], [1, 8])
    mark(d, [z], "cand")
    mark(d, [(3, 4)], "alias")
    d["labels"][key(z)] = "Z"
    for k, b in seat.items():
        d["arrows"].append([b, k, "blocks-z" if b == z else "blocks"])
    d["arrows"].append([(3, 4), (4, 5), "blocks-z"])
    d["arrows"].append([z, (3, 4), "link"])
    d["blocks"].append([3, 5, "dead"])
    d["pixels"].append([4, 6, "dead"])
    F["refute_defect"] = d
    facts.update(ref_z=z, ref_first=e["first"], ref_chain=e["chain"], ref_seats=e["seats"], ref_why=e["why"])
    return F, facts


def main():
    F, facts = build()
    for k, v in facts.items():
        print(k, v)
    tpl = open(os.path.join(HERE, "foundation_template.html"), encoding="utf-8").read()
    shown = {k: facts[k] for k in ("n_terms", "with_p", "with_diag", "own", "defects",
                                    "n_newly_blocked", "n_links_before", "n_links_after")}
    out = tpl.replace("/*FIGURES*/null", json.dumps(F)).replace("/*FACTS*/null", json.dumps(shown))
    open(os.path.join(HERE, "foundation.html"), "w", encoding="utf-8").write(out)
    print("written", os.path.join(HERE, "foundation.html"))


if __name__ == "__main__":
    main()
