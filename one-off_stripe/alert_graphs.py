from map_of_squares import StateEnum, InvalidTilingError

# The 8 neighbours of a cell (direct and diagonal), in ring order so that RING[k-1]
# and RING[k+1] (mod 8) are the two neighbours geometrically adjacent to RING[k].
RING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


# Each of the four 2x2 blocks touching a cell is made up of that cell plus three ring
# neighbours: two direct ones and the diagonal between them - here as ring-index
# triples.
QUADRANT_TRIPLES = [(7, 0, 1), (1, 2, 3), (3, 4, 5), (5, 6, 7)]


def iter_alert_thirds(ring):
    """Yield the ring index of the free corner completing each real "alert" 2x2 block
    around the centre. Each QUADRANT_TRIPLE (direct, corner, direct) is one such
    block; it becomes an alert once the centre joins two already-blocked members of
    the triple, leaving the third free. Two of the triple's three members already
    blocked can happen two ways:

    - a ring-adjacent pair, direct+corner or corner+direct, blocked - leaving the
      triple's other direct neighbour as the free third corner; or
    - the triple's two direct neighbours both blocked, with the corner between them
      (ring-non-adjacent to either) still free - leaving that corner as the free
      third corner.

    A blocked pair whose third corner is already blocked (or chosen) is not an
    alert - blocking the centre there would not produce a three-blocked-one-free
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
    """Raise alert_blocked on item if blocking it would risk an "alert" situation:
    a 2x2 block with three items blocked and one (this item) free.
    """
    for _ in iter_alert_thirds(ring):
        item.alert_blocked = True
        break


def set_alert_chosen(item, i, j, ring):
    """For an item with alert_blocked set, blocking it would put some 2x2 block at
    three-blocked-one-free - the "alert" situation. Find that block's remaining free
    corner among ring, raise its alert_chosen flag, and append it to item.reverse_links so
    the alert_blocked item knows which item(s) resolve its risk - and append item's
    own position to that corner's .links, the matching incoming-side record
    (see SquareItem.links).

    A single item can have more than one such block (iter_alert_thirds can yield
    several thirds) - every one of them ends up in item.reverse_links (see
    SquareItem.reverse_links), so an item at risk from two different quadrants at once
    still has both recorded. But two different quadrants can share the same free
    completing corner (e.g. one ring position can be the "third" for both its
    ring-adjacent triples at once - see test_do_closure_steps, where (6, 2)'s
    corner at ring index 1 completes both triple (1, 2, 3) and triple (7, 0, 1)),
    so iter_alert_thirds can yield the same ring index twice. Recording each
    yield into item.reverse_links directly would append that same position twice for one
    real pairing - alert_chosen_local (one bool per ring position) collapses
    that: every yield just flags its ring index, however many times it's
    yielded, and only after the loop is each flagged position turned into a
    single reverse_links entry.
    """
    alert_chosen_local = [False] * 8
    for third_idx in iter_alert_thirds(ring):
        ring[third_idx].alert_chosen = True
        alert_chosen_local[third_idx] = True

    for third_idx in range(8):
        if alert_chosen_local[third_idx]:
            di, dj = RING_OFFSETS[third_idx]
            item.reverse_links.append((i + di, j + dj))
            ring[third_idx].links.append((i, j))


# The four diagonal neighbours - the ones place_square_in_core blocks when an item is
# chosen - as a subset of RING_OFFSETS (ring indices 0, 2, 4, 6).
DIAGONAL_OFFSETS = [RING_OFFSETS[k] for k in (0, 2, 4, 6)]


def resolve_chosen_link(i, j, map_of_squares):
    """Choosing an alert_chosen item blocks its four diagonal neighbours. Every one
    of them that is itself alert_blocked already has a link to the alert_chosen
    item that would resolve its own risk - return all such reverse_links (see
    SquareItem.reverse_links), so the alert_chosen items involved all end up linked to
    (i, j): choosing it obliges choosing every one of them too.

    Returns every qualifying link, in DIAGONAL_OFFSETS order, as a list (empty if
    none qualify) - not just the last one found. An item can have as many as
    four diagonal neighbours that separately qualify (see test_4links_situation,
    where one item's diagonal neighbours are each alert_blocked with a different
    link), and every one of those is a real, independent consequence of choosing
    (i, j) - keeping only one and dropping the rest would silently lose the
    others' obligations, exactly the "crowding out" .reverse_links exists to avoid.

    Only an individual link entry that points straight back at (i, j) is
    skipped, not the whole neighbour: that particular entry just reflects a
    mutual pairing this item and that neighbour already have straight out of
    find_alerts (each is the other's alert-completing corner for that one
    quadrant - see iter_alert_thirds), and including it here would only add
    (i, j) itself, a self-loop that says nothing (choosing this item does not,
    by itself, oblige choosing itself again). But an alert_blocked neighbour can
    have more than one entry in its own .reverse_links (it can be at risk from more than
    one quadrant at once), and any OTHER entry is a genuine, separate obligation
    that must still be picked up - checking only the neighbour's first link
    would wrongly discard a real relay just because that neighbour's first
    entry happens to be the self-loop (see
    test_link_patches_relays_past_self_loop_candidate in test_representation.py:
    (2, 7)'s reverse_links are [(1, 6), (1, 8)] - the first is the self-loop back to
    (1, 6), but the second, to (1, 8), is exactly the relay that must survive).

    A pure read over map_of_squares - does not mutate anything. A diagonal
    neighbour can itself be an alert_chosen item whose own .reverse_links this same stage
    is about to append to, so callers must resolve every item's new reverse_links from
    one consistent snapshot and only apply the results afterwards (see
    link_patches) - otherwise the answer depends on which item happens to be
    visited first.
    """
    reverse_links = []
    for di, dj in DIAGONAL_OFFSETS:
        neighbour = map_of_squares[i + di, j + dj]
        if not neighbour.alert_blocked:
            continue
        for candidate in neighbour.reverse_links:
            if candidate != (i, j) and candidate not in reverse_links:
                reverse_links.append(candidate)
    return reverse_links


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
