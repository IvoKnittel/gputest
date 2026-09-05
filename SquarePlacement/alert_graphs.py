from map_of_squares import StateEnum

# The 8 neighbours of a cell (direct and diagonal), in ring order so that RING[k-1]
# and RING[k+1] (mod 8) are the two neighbours geometrically adjacent to RING[k].
RING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


# Each of the four 2x2 blocks touching a cell is made up of that cell plus three ring
# neighbours: two direct ones and the diagonal between them - here as ring-index
# triples.
QUADRANT_TRIPLES = [(7, 0, 1), (1, 2, 3), (3, 4, 5), (5, 6, 7)]


def iter_alert_thirds(ring):
    """Yield the ring index of the free corner completing each real seat around the
    centre - team term for what this module's own identifiers (alert_blocked,
    alert_chosen, find_alerts_set_links) call an "alert": a 2x2 block with three corners
    blocked and one free, the free one being the seat an alert_chosen item must
    fill (see docs/rose_cascades_and_holes/README.md). Each QUADRANT_TRIPLE
    (direct, corner, direct) is one such block; it becomes a seat once the centre
    joins two already-blocked members of the triple, leaving the third free. Two of
    the triple's three members already blocked can happen two ways:

    - a ring-adjacent pair, direct+corner or corner+direct, blocked - leaving the
      triple's other direct neighbour as the free third corner; or
    - the triple's two direct neighbours both blocked, with the corner between them
      (ring-non-adjacent to either) still free - leaving that corner as the free
      third corner.

    A blocked pair whose third corner is already blocked (or chosen) is not a
    seat - blocking the centre there would not produce a three-blocked-one-free
    block, so it is skipped.
    """
    for idx in range(8):
        if ring[idx].state != StateEnum.blocked or ring[(idx + 1) % 8].state != StateEnum.blocked:  # BR-032
            continue
        triple = next(t for t in QUADRANT_TRIPLES if idx in t and (idx + 1) % 8 in t)
        third_idx = next(k for k in triple if k not in (idx, (idx + 1) % 8))
        if ring[third_idx].state == StateEnum.free:  # BR-033
            yield third_idx

    for direct_a, corner, direct_b in QUADRANT_TRIPLES:
        if (ring[direct_a].state == StateEnum.blocked
                and ring[direct_b].state == StateEnum.blocked
                and ring[corner].state == StateEnum.free):  # BR-034
            yield corner


def set_alert_blocked(item, ring):
    """Raise alert_blocked on item if blocking it would risk completing a seat:
    a 2x2 block with three items blocked and one (this item) free.
    """
    for _ in iter_alert_thirds(ring):
        item.alert_blocked = True
        break


def set_alert_chosen_set_links(i, j, ring):
    """For a centre at (i, j) with alert_blocked set, blocking it would put some
    2x2 block at three-blocked-one-free - a seat. Find that seat's remaining free
    corner(s) among ring and raise their alert_chosen flag - so far, as before.

    The link this creates is NOT between (i, j) and the corner. (i, j) itself is
    never going to be *chosen* in this scenario - it's the item at risk of being
    *blocked*, and .forces means "if this gets chosen, these must too" everywhere
    else it's read (forced_closure); "if this gets blocked"
    is a different event entirely, one (i, j) being chosen has nothing to do with.
    What actually forces a corner is whichever of (i, j)'s own diagonal neighbours,
    if chosen, would block (i, j) as a side effect - so it's each free diagonal
    ring position (not (i, j) itself) that gets linked to the corner(s) it would
    end up forcing. Any one of (i, j)'s four diagonal neighbours blocks it just the
    same, regardless of which quadrant originally made (i, j) alert_blocked - so
    every diagonal ring position that's currently free gets a link to every corner
    found here, not only the one diagonal neighbour geometrically tied to one
    specific quadrant. (This is also what makes a later, separate relay stage for
    "any free item diagonal to an alert_blocked item" unnecessary.) A diagonal ring position that happens to
    itself be one of the corners found is excluded from its own link (no self-loop).

    A single centre can have more than one corner (iter_alert_thirds can yield
    several) - alert_chosen_local (one bool per ring position) collapses a corner
    independently found twice (e.g. one ring position completing two different
    triples at once) into a single entry.
    """
    alert_chosen_local = [False] * 8
    for third_idx in iter_alert_thirds(ring):
        ring[third_idx].alert_chosen = True
        alert_chosen_local[third_idx] = True

    corner_indices = [k for k in range(8) if alert_chosen_local[k]]

    for d_idx in (0, 2, 4, 6):
        diagonal_item = ring[d_idx]
        if diagonal_item.state != StateEnum.free:  # BR-035
            continue
        di, dj = RING_OFFSETS[d_idx]
        d_pos = (i + di, j + dj)
        for c_idx in corner_indices:
            if c_idx == d_idx:  # BR-036
                continue
            ci, cj = RING_OFFSETS[c_idx]
            c_pos = (i + ci, j + cj)
            diagonal_item.forces.add(c_pos)
            ring[c_idx].forced_by.add(d_pos)


# The four diagonal neighbours - the ones a chosen item blocks (see closure.place_squares)
# - as a subset of RING_OFFSETS (ring indices 0, 2, 4, 6).
DIAGONAL_OFFSETS = [RING_OFFSETS[k] for k in (0, 2, 4, 6)]


