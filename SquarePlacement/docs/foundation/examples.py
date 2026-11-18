"""The foundation document's examples, computed with engine.py (vacuum
conditions + compute_conditions only). Run directly to print the facts the
prose relies on; make_foundation.py turns them into figures."""
import itertools

from engine import (board, choose, settle, links, chain, simulate, states, Impossible,
                    initial_map, vacuum_condition_offsets, compute_conditions, CHOSEN, BLOCKED, FREE,
                    diagonals, blocking_pairs)

PINWHEEL_BLOCK = ((2, 2), (2, 3), (3, 2), (3, 3))      # the four cells around pixel (3,3)
PINWHEEL = {(2, 2): (3, 1), (2, 3): (1, 2), (3, 2): (4, 3), (3, 3): (2, 4)}   # corner -> its blocker


def terms(m, y):
    """y's conditions as sorted lists of member ids."""
    return sorted(sorted(p for p, _s in t) for t in m[y].forced_by)


def defect_count():
    D = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    block = [(0, 0), (0, 1), (1, 0), (1, 1)]
    per = [[(c[0] + a, c[1] + b) for a, b in D if (c[0] + a, c[1] + b) not in block] for c in block]
    return len({frozenset(t) for t in itertools.product(*per)
                if not any(abs(p[0] - q[0]) == 1 and abs(p[1] - q[1]) == 1
                           for p, q in itertools.combinations(t, 2))})


def seat_of(m, a, y):
    """The 2x2 block around y whose three other corners are blocked once a is
    chosen (in m's state), with each corner's blocker: {corner: blocker}."""
    for dr, dc in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
        corners = [(y[0] + dr + i, y[1] + dc + j) for i in (0, 1) for j in (0, 1)]
        others = [k for k in corners if k != y]
        blockers = {}
        for k in others:
            if abs(k[0] - a[0]) == 1 and abs(k[1] - a[1]) == 1:
                blockers[k] = a
            elif m[k].state == BLOCKED:
                blockers[k] = next(q for q in diagonals(m, k) if m[q].state == CHOSEN)
        if len(blockers) == 3:
            return blockers
    return None


def ex_conditions():
    m = initial_map((7, 7))
    y = (3, 3)
    t = sorted(PINWHEEL[k] for k in PINWHEEL_BLOCK if k != y)
    assert t in terms(m, y)
    return dict(m=m, y=y, n_terms=len(m[y].forced_by), term=t, n_vacuum=len(vacuum_condition_offsets()))


def ex_place():
    m = initial_map((9, 9))
    p = (4, 4)
    cells = [(i, j) for i in range(9) for j in range(9)]
    with_p = sum(1 for c in cells for t in m[c].forced_by if any(q == p for q, _ in t))
    diag = diagonals(m, p)
    with_diag = sum(1 for c in cells for t in m[c].forced_by if any(q in diag for q, _ in t))
    own = len(m[p].forced_by) + sum(len(m[q].forced_by) for q in diag)
    choose(m, p)
    before = {c: set(m[c].forced_by) for c in cells}
    compute_conditions(m)
    shrunk = sum(1 for c in cells for t in m[c].forced_by if len(t) == 2)
    return dict(m=m, p=p, diag=diag, with_p=with_p, with_diag=with_diag, own=own, pairs_after=shrunk)


def ex_seat():
    """Choose the pinwheel blockers of (2,2), (2,3), (3,2) one by one: the
    condition {(1,2),(3,1),(4,3)} of (3,3) shrinks to a pair, a link, and
    (3,3) is forced."""
    frames = []
    m = initial_map((7, 7))
    order = [(1, 2), (3, 1), (4, 3)]
    full = sorted(order)
    frames.append(dict(st=states(m), chosen=[], term=full, links=[]))
    for k, p in enumerate(order):
        choose(m, p)
        forced = compute_conditions(m)
        rest = sorted(q for q in order[k + 1:])
        frames.append(dict(st=states(m), chosen=order[:k + 1], term=rest, forced=forced,
                           links=sorted(links(m))))
        for f in forced:
            choose(m, f)
        settle(m)
    frames[-1]["after"] = states(m)
    return frames


def ex_path():
    m = board((11, 11), [(2, 7), (5, 7), (8, 6)])
    z = (5, 4)
    L = links(m)
    c = chain(m, z, L)
    inner = sorted((a, y) for a, y in L if a in c and y in c)
    m2, forced = simulate(m, z)
    assert set(forced) == c - {z}
    return dict(m=m, z=z, chain=sorted(c), links=inner, after=m2, forced=sorted(forced),
                seats={(a, y): seat_of(m, a, y) for a, y in inner})


def ex_pair():
    m = board((11, 11), [(4, 5), (7, 5), (8, 7)])
    z1, z2 = (4, 2), (7, 2)
    L = links(m)
    c1, c2 = chain(m, z1, L), chain(m, z2, L)
    for z, c in ((z1, c1), (z2, c2)):
        _m, f = simulate(m, z)
        assert set(f) == c - {z} and not blocking_pairs(c)
    pairs = sorted((x, y) for x in c1 for y in c2 if abs(x[0] - y[0]) == 1 and abs(x[1] - y[1]) == 1)
    seats = {}
    for a, y in L:
        if (a in c1 and y in c1) or (a in c2 and y in c2):
            seats[y] = (a, seat_of(m, a, y))
    return dict(m=m, z1=z1, z2=z2, c1=sorted(c1), c2=sorted(c2), pairs=pairs, seats=seats,
                links=sorted((a, y) for a, y in L if a in c1 | c2))


def ex_refute():
    m = board((10, 10), [(2, 4), (2, 7), (5, 2)])
    z = (5, 5)
    L = links(m)
    first = sorted(y for a, y in L if a == z)
    c = chain(m, z, L)
    seats = {y: seat_of(m, z, y) for y in first}
    try:
        simulate(m, z)
        raise AssertionError("expected a contradiction")
    except Impossible as e:
        why = str(e)
    return dict(m=m, z=z, first=first, first_pairs=blocking_pairs(set(first) | {z}),
                chain=sorted(c), chain_pairs=blocking_pairs(c), seats=seats, why=why,
                links=sorted((a, y) for a, y in L if a in c and y in c))


if __name__ == "__main__":
    e = ex_conditions()
    print("conditions:", e["n_terms"], "terms of", e["y"], "; vacuum", e["n_vacuum"], "; example", e["term"])
    print("defect terms per block:", defect_count())
    e = ex_place()
    print("place", e["p"], ": terms with p as member", e["with_p"], "; with a diagonal as member", e["with_diag"],
          "; own+diagonals' terms", e["own"], "; pairs after", e["pairs_after"])
    for f in ex_seat():
        print("seat frame:", f["chosen"], "term", f["term"], "forced", f.get("forced"),
              "links into (3,3):", [l for l in f["links"] if l[1] == (3, 3)])
    e = ex_path()
    print("path:", e["z"], e["chain"], e["links"], "seats", e["seats"])
    e = ex_pair()
    print("pair:", e["c1"], e["c2"], e["pairs"], "seats", e["seats"])
    e = ex_refute()
    print("refute:", e["first"], e["first_pairs"], "| chain", e["chain"], e["chain_pairs"], "| seats", e["seats"],
          "|", e["why"])
