"""A small engine for the foundation document's examples - only the vacuum
conditions (conditions.initial_map) and their reduction
(bruteforce.compute_conditions), nothing of the pyramid or the n-ary step.

choose(m, p): p chosen, its four diagonal neighbours blocked.
settle(m): place every forced cell until nothing more is forced.
links(m): the single-member conditions, as {(a, y)}: a -> y.
chain(m, z): z and every cell z reaches over the links (its aliases).
simulate(m, z): a copy of m with z chosen and settled; returns (copy, forced).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from conditions import initial_map, vacuum_condition_offsets   # noqa: E402
from bruteforce import compute_conditions                        # noqa: E402
from map_of_squares import StateEnum                             # noqa: E402

DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
FREE, CHOSEN, BLOCKED = StateEnum.free, StateEnum.chosen, StateEnum.blocked


class Impossible(Exception):
    pass


def on_board(m, p):
    return 0 <= p[0] < m.shape[0] and 0 <= p[1] < m.shape[1]


def diagonals(m, p):
    return [(p[0] + a, p[1] + b) for a, b in DIAG if on_board(m, (p[0] + a, p[1] + b))]


def choose(m, p):
    if m[p].state == BLOCKED:
        raise Impossible(f"{p} is blocked")
    m[p].state = CHOSEN
    for q in diagonals(m, p):
        if m[q].state == CHOSEN:
            raise Impossible(f"{p} and {q} are chosen diagonal neighbours")
        m[q].state = BLOCKED


def settle(m):
    placed = []
    while True:
        forced = compute_conditions(m)
        if not forced:
            return placed
        for f in forced:
            choose(m, f)
            placed.append(f)


def board(shape, chosen=()):
    m = initial_map(shape)
    for p in chosen:
        choose(m, p)
    settle(m)
    return m


def links(m):
    out = set()
    for y in zip(*[a.ravel() for a in __import__("numpy").indices(m.shape)]):
        y = (int(y[0]), int(y[1]))
        if m[y].state != FREE:
            continue
        for cond in m[y].forced_by:
            if len(cond) == 1:
                (a, _s), = cond
                out.add((a, y))
    return out


def chain(m, z, link_set=None):
    link_set = links(m) if link_set is None else link_set
    fwd = {}
    for a, y in link_set:
        fwd.setdefault(a, set()).add(y)
    seen, todo = {z}, [z]
    while todo:
        for y in fwd.get(todo.pop(), ()):
            if y not in seen:
                seen.add(y)
                todo.append(y)
    return seen


def simulate(m, z):
    m2 = copy.deepcopy(m)
    choose(m2, z)
    return m2, settle(m2)


def states(m):
    return [[{FREE: 0, CHOSEN: 1, BLOCKED: 2}[m[i, j].state] for j in range(m.shape[1])]
            for i in range(m.shape[0])]


def blocking_pairs(cells):
    cells = set(cells)
    return sorted((x, (x[0] + 1, x[1] + dc)) for x in cells for dc in (-1, 1)
                  if (x[0] + 1, x[1] + dc) in cells)
