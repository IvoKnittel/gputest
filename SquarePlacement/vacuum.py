"""Measures how .forced_by behaves on a pure vacuum board (nothing chosen or
blocked anywhere) as the realm you look at grows from one base tile up to the
whole board - see squareplacement_superlattice_refactor memory note, "face the
problem without compromise first."

Nothing can be discarded in a vacuum (bruteforce.compute_conditions only ever drops a
condition once some cell becomes chosen or blocked - see conditions.py), so
every condition is still a full triple everywhere. The question this asks is
how the *boundary-crossing* conditions of one realm - the ones a CoreItem
would have to carry up because at least one triple member reaches outside it,
see cores.py's in_realm/CoreItem.update_dependencies - behave as that realm
grows.

The realm sizes (3, 9, 27, 81) follow cores.py's own pyramid exactly - each
level is a 3x3 block of the level below, not an arbitrary bigger window - with
81x81 chosen as the board size itself, so that level is genuinely "the whole
board," not just "a big realm." All four are centered on the same point,
with enough clearance to the board edge (27 cells on every side, well past
the radius-2 reach of a vacuum condition offset) that none of them is
truncated by the board boundary instead of genuinely resolving.

This does not yet correspond to one of image_to_squares.py's actual
placement tiles (core_range_for_tile/tile_upper_left_indices starts cores at
specific phase-offset positions, wrapped in a 1-cell margin, not margin-3) -
these are plain concentric windows, picked only to measure the pyramid's own
scaling behaviour in isolation.
"""
from collections import Counter

from conditions import initial_map
from cores import in_realm


def realm_centered(size, center):
    """A size x size realm (see cores.in_realm's own shape), centered on
    center - odd size assumed, same convention as the rest of this module."""
    r0 = center[0] - size // 2
    c0 = center[1] - size // 2
    return (r0, r0 + size), (c0, c0 + size)


def boundary_crossing(m, realm):
    """Every (cell, condition) pair where cell lies inside realm and
    condition (one of cell's own .forced_by triples) has at least one member
    outside realm - i.e. exactly what a CoreItem covering realm would still
    have to carry upward, since bruteforce.compute_conditions/
    cores.CoreItem.update_dependencies both only ever drop a pair/condition
    once every member lies inside one realm.
    """
    (r0, r1), (c0, c1) = realm
    crossing = []
    for i in range(r0, r1):
        for j in range(c0, c1):
            for condition in m[i, j].forced_by:
                if not all(in_realm(id_, realm) for id_, _status in condition):
                    crossing.append(((i, j), condition))
    return crossing


def external_id_counts(crossing, realm):
    """How many of the boundary-crossing (cell, condition) pairs each
    distinct external id (one lying outside realm) appears in - the
    entanglement/reuse measure: a high count means that id's condition can't
    be treated as an independent fact, it's shared by many cells at once."""
    counts = Counter()
    for _cell, condition in crossing:
        for id_, _status in condition:
            if not in_realm(id_, realm):
                counts[id_] += 1
    return counts


SIZE = (81, 81)
CENTER = (40, 40)


def test_vacuum_conditions_never_shrink_to_links():
    """At the base 3x3 tile, every single condition is still a full,
    unsimplified triple - none has been reduced to a pair/link, because
    nothing has been chosen or blocked to discard anything against."""
    m = initial_map(SIZE)
    realm = realm_centered(3, CENTER)
    crossing = boundary_crossing(m, realm)
    assert len(crossing) == 684
    assert all(len(condition) == 3 for _cell, condition in crossing)


def test_vacuum_boundary_crossing_grows_then_collapses():
    """Climbing cores.py's own 3x3-per-level pyramid on a vacuum board makes
    the carried-up set bigger at every intermediate level, not smaller - more
    cells, each still fully unresolved, come into view at each step - until
    the realm becomes the whole board, where there is no "outside" left and
    the set collapses to exactly nothing."""
    m = initial_map(SIZE)
    counts = {size: len(boundary_crossing(m, realm_centered(size, CENTER)))
              for size in (3, 9, 27, 81)}
    assert counts[3] == 684
    assert counts[9] == 3084
    assert counts[27] == 10284
    assert counts[9] > counts[3]
    assert counts[27] > counts[9]
    assert counts[81] == 0


def test_vacuum_external_ids_are_heavily_reused():
    """The boundary-crossing conditions at the base tile aren't independent
    facts sitting side by side - a handful of external cells are each a
    member of dozens of different conditions, so merging overlapping
    conditions (rather than discarding them independently) would produce much
    bigger joint terms than a plain triple."""
    m = initial_map(SIZE)
    realm = realm_centered(3, CENTER)
    crossing = boundary_crossing(m, realm)
    counts = external_id_counts(crossing, realm)
    assert len(counts) == 40
    assert max(counts.values()) == 57
    assert sum(1 for v in counts.values() if v > 1) == 40
