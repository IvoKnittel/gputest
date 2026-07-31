from map_of_squares import StateEnum, InvalidTilingError

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
    alert_chosen, find_alerts) call an "alert": a 2x2 block with three corners
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
        if ring[idx].state != StateEnum.blocked or ring[(idx + 1) % 8].state != StateEnum.blocked:
            continue
        triple = next(t for t in QUADRANT_TRIPLES if idx in t and (idx + 1) % 8 in t)
        third_idx = next(k for k in triple if k not in (idx, (idx + 1) % 8))
        if ring[third_idx].state == StateEnum.free:
            yield third_idx

    for direct_a, corner, direct_b in QUADRANT_TRIPLES:
        if (ring[direct_a].state == StateEnum.blocked
                and ring[direct_b].state == StateEnum.blocked
                and ring[corner].state == StateEnum.free):
            yield corner


def set_alert_blocked(item, ring):
    """Raise alert_blocked on item if blocking it would risk completing a seat:
    a 2x2 block with three items blocked and one (this item) free.
    """
    for _ in iter_alert_thirds(ring):
        item.alert_blocked = True
        break


def set_alert_chosen(i, j, ring):
    """For a centre at (i, j) with alert_blocked set, blocking it would put some
    2x2 block at three-blocked-one-free - a seat. Find that seat's remaining free
    corner(s) among ring and raise their alert_chosen flag - so far, as before.

    The link this creates is NOT between (i, j) and the corner. (i, j) itself is
    never going to be *chosen* in this scenario - it's the item at risk of being
    *blocked*, and .forces means "if this gets chosen, these must too" everywhere
    else it's read (forced_closure, resolve_chosen_link); "if this gets blocked"
    is a different event entirely, one (i, j) being chosen has nothing to do with.
    What actually forces a corner is whichever of (i, j)'s own diagonal neighbours,
    if chosen, would block (i, j) as a side effect - so it's each free diagonal
    ring position (not (i, j) itself) that gets linked to the corner(s) it would
    end up forcing. Any one of (i, j)'s four diagonal neighbours blocks it just the
    same, regardless of which quadrant originally made (i, j) alert_blocked - so
    every diagonal ring position that's currently free gets a link to every corner
    found here, not only the one diagonal neighbour geometrically tied to one
    specific quadrant. (This is also what makes a later, separate relay stage for
    "any free item diagonal to an alert_blocked item" unnecessary - see
    resolve_chosen_link/link_patches.) A diagonal ring position that happens to
    itself be one of the corners found is excluded from its own link (no self-loop).

    A single centre can have more than one corner (iter_alert_thirds can yield
    several) - alert_chosen_local (one bool per ring position) collapses a corner
    independently found twice (see test_do_closure_steps, where (6, 2)'s ring
    index 1 completes two different triples at once) into a single entry, same
    as before.
    """
    alert_chosen_local = [False] * 8
    for third_idx in iter_alert_thirds(ring):
        ring[third_idx].alert_chosen = True
        alert_chosen_local[third_idx] = True

    corner_indices = [k for k in range(8) if alert_chosen_local[k]]

    for d_idx in (0, 2, 4, 6):
        diagonal_item = ring[d_idx]
        if diagonal_item.state != StateEnum.free:
            continue
        di, dj = RING_OFFSETS[d_idx]
        d_pos = (i + di, j + dj)
        for c_idx in corner_indices:
            if c_idx == d_idx:
                continue
            ci, cj = RING_OFFSETS[c_idx]
            c_pos = (i + ci, j + cj)
            diagonal_item.forces.append(c_pos)
            ring[c_idx].forced_by.append(d_pos)


# The four diagonal neighbours - the ones place_square_in_core blocks when an item is
# chosen - as a subset of RING_OFFSETS (ring indices 0, 2, 4, 6).
DIAGONAL_OFFSETS = [RING_OFFSETS[k] for k in (0, 2, 4, 6)]


def resolve_chosen_link(i, j, map_of_squares):
    """Formerly stage 3's relay: for every diagonal neighbour of (i, j) that is
    alert_blocked, read its .forces to find what choosing (i, j) - which blocks
    that neighbour - would additionally oblige.

    That premise no longer holds now that set_alert_chosen links every free
    diagonal neighbour of an alert_blocked centre directly, not the centre
    itself (see its docstring): an alert_blocked neighbour's own .forces no
    longer describes "what to do if I get blocked" - it describes whatever
    *that neighbour* separately forces as someone else's diagonal linker, which
    has nothing to do with (i, j)'s own risk. Reading it here doesn't find a
    deeper relay anymore, it finds an unrelated value and wrongly overwrites
    (i, j)'s own already-correct forces with it (verified directly: in the
    (3,3)/(4,4) cascade from test_rose_cascades_and_holes.py, set_alert_chosen
    alone already gives (3,3).forces == [(4,5)] - correct - and the old version
    of this function replaced it with [(2,3)], which (3,3) has no actual path
    to).

    Whatever this function used to contribute - including the exact case it was
    built for (test_4links_situation: an item with several alert_blocked
    diagonal neighbours, none of which happens to be the literal corner of any
    one of their own quadrants) - set_alert_chosen now establishes directly and
    correctly in a single pass, and closure.forced_closure already provides the
    transitive, multi-hop walk over that corrected data - a relay stage on top
    of it is not just unneeded, it actively corrupts correct values wherever it
    used to fire. So there is nothing left for this stage to find.

    Kept as a named no-op, rather than deleted, so link_patches and every
    caller of it keep working unchanged - it is now guaranteed to return [].
    """
    return []


def mark_patch_conflicts(item, i, j, map_of_squares):
    """Choosing an alert_chosen item blocks its four diagonal neighbours. If one of
    them is itself alert_chosen with a different patch_id, choosing either patch would
    block a member of the other, so the two patches exclude each other - record each
    other's patch_id in .conflicts (skipping duplicates). If one of them shares this
    item's own patch_id, the patch would end up blocking one of its own members, which
    should never happen - raise InvalidTilingError.
    """
    for di, dj in DIAGONAL_OFFSETS:
        neighbour = map_of_squares[i + di, j + dj]
        if not neighbour.alert_chosen:
            continue
        if neighbour.patch_id == item.patch_id:
            raise InvalidTilingError(
                f"patch {item.patch_id} is self-contradictory: item at "
                f"({i}, {j}) would block its own member at ({i + di}, {j + dj})")
        if neighbour.patch_id not in item.conflicts:
            item.conflicts.append(neighbour.patch_id)
        if item.patch_id not in neighbour.conflicts:
            neighbour.conflicts.append(item.patch_id)
