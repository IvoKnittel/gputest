from map_of_squares import StateEnum

# The 8 neighbours of a cell (direct and diagonal), in ring order so that RING[k-1]
# and RING[k+1] (mod 8) are the two neighbours geometrically adjacent to RING[k].
RING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


# Each of the four 2x2 blocks touching a cell is made up of that cell plus three ring
# neighbours: two direct ones and the diagonal between them - here as ring-index
# triples.
QUADRANT_TRIPLES = [(7, 0, 1), (1, 2, 3), (3, 4, 5), (5, 6, 7)]


def iter_alert_thirds(ring):
    """Yield the ring index of the free corner completing each real "alert" 2x2 block
    around the centre: two ring-adjacent blocked neighbours plus a still-free third
    corner. A blocked ring-adjacent pair whose third corner is already blocked (or
    chosen) is not an alert - blocking the centre there would not produce a
    three-blocked-one-free block, so it is skipped.
    """
    for idx in range(8):
        if ring[idx].state != StateEnum.blocked or ring[(idx + 1) % 8].state != StateEnum.blocked:
            continue
        triple = next(t for t in QUADRANT_TRIPLES if idx in t and (idx + 1) % 8 in t)
        third_idx = next(k for k in triple if k not in (idx, (idx + 1) % 8))
        if ring[third_idx].state == StateEnum.free:
            yield third_idx


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
    corner among ring, raise its alert_chosen flag, and point item.link at its index
    so the alert_blocked item knows which item resolves its risk.
    """
    for third_idx in iter_alert_thirds(ring):
        ring[third_idx].alert_chosen = True
        di, dj = RING_OFFSETS[third_idx]
        item.link = (i + di, j + dj)


# The four diagonal neighbours - the ones place_square_in_core blocks when an item is
# chosen - as a subset of RING_OFFSETS (ring indices 0, 2, 4, 6).
DIAGONAL_OFFSETS = [RING_OFFSETS[k] for k in (0, 2, 4, 6)]


def link_chosen_items(item, i, j, map_of_squares):
    """Choosing an alert_chosen item blocks its four diagonal neighbours. If one of
    them is itself alert_blocked, it already has a link to the alert_chosen item that
    would resolve its own risk - adopt that link here, so the two alert_chosen items
    end up linked to each other: choosing one obliges choosing the other too.
    """
    for di, dj in DIAGONAL_OFFSETS:
        neighbour = map_of_squares[i + di, j + dj]
        if neighbour.alert_blocked and neighbour.link is not None:
            item.link = neighbour.link


def assign_graph_id(item, i, j, rows, map_of_squares):
    """Give item a graph_id if it doesn't have one yet (using its flattened index
    i * rows + j), then reconcile it with its linked item's graph_id: if the linked
    item has none yet, it adopts item's id; if both already have one, the smaller of
    the two wins and is assigned to both. This way a chain of linked alert_chosen
    items converges on a single shared graph_id regardless of visit order.
    """
    if item.graph_id == -1:
        item.graph_id = i * rows + j
    if item.link is not None:
        linked = map_of_squares[item.link]
        if linked.graph_id == -1:
            linked.graph_id = item.graph_id
        else:
            shared_id = min(item.graph_id, linked.graph_id)
            item.graph_id = shared_id
            linked.graph_id = shared_id
