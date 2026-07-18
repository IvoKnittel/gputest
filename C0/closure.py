from map_of_squares import StateEnum, InvalidTilingError
from alert_graphs import (RING_OFFSETS,
                           set_alert_blocked,
                           set_alert_chosen,
                           link_chosen_items,
                           assign_graph_id,
                           mark_graph_conflicts as mark_item_graph_conflicts)


def find_graphs(map_of_squares):
    """Discover alert_chosen items, link them, and group them into graphs.

    -- Data model (SquareItem / StateEnum) --
    map_of_squares is a padded 2D object array of SquareItem, one per block-position.
    Each item has:
      .quality        candidate score (from image_squares_ranked); -1 in the padding.
      .state          StateEnum.free / .chosen / .blocked.
      .alert_blocked  see stage 2 below.
      .alert_chosen   see stage 2 below.
      .link           (row, col) index of another item this one is paired with, or
                       None - see stages 2 and 3.
    Padding cells start out permanently StateEnum.blocked (quality < 0), so they can
    never be selected and the loops below only need a 1-cell margin of safety.

    -- Parallel placement, for context (place_square_in_core / test_square_placement
    / test_tiling) --
    The plane is covered by non-overlapping 3x3 cores on a 5-wide tile grid (1-cell
    border on each side). Only one sublattice of tiles - chosen by (row parity, col
    parity), 4 sublattices total - is active in any single simulated CUDA launch,
    because active tiles within one sublattice are 2 tiles (6 core-cells) apart, so no
    active core, nor its 1-cell diagonal write, ever touches another active tile's
    cells in the same pass. That is what lets place_square_in_core write
    .state = chosen/blocked directly, without needing to serialize across tiles.

    -- What find_graphs resolves after a sweep --
    Direct writes keep the 4 sublattice passes conflict-free, but afterwards the plane
    can still be in a state no single placement call could safely react to by itself.
    find_graphs runs sequentially over the whole map to resolve that, in five stages,
    still written with that parallel/sequential split in mind:

    1. Promote any free cell whose 4 direct (orthogonal) neighbours are all blocked to
       chosen. Such a cell can never overlap a chosen square, so placing it is always
       safe - and it is the only way it will ever get covered, since no future core
       sweep would treat it as reachable.

    2. alert_blocked / alert_chosen (set_alert_blocked, set_alert_chosen,
       iter_alert_thirds): an "alert" is a 2x2 block with three items blocked and one
       free. For a free item, look at its 8 neighbours (direct + diagonal) in ring
       order (RING_OFFSETS). Each of the 4 possible 2x2 blocks touching the item
       corresponds to one run of 3 consecutive ring indices (QUADRANT_TRIPLES): two
       direct neighbours and the diagonal between them. If two ring-adjacent
       neighbours are blocked *and* the third (completing) corner of that block is
       still free, blocking this item would turn that block into a real alert -
       iter_alert_thirds yields exactly those completing corners. When that happens:
       set_alert_blocked raises .alert_blocked on the item under consideration, and
       set_alert_chosen raises .alert_chosen on the free completing corner, recording
       the pairing by pointing the alert_blocked item's .link at the alert_chosen
       item's index.

    3. link_chosen_items: choosing an item blocks its 4 diagonal neighbours
       (DIAGONAL_OFFSETS). For every alert_chosen item, if one of those diagonal
       neighbours is itself alert_blocked (and already has a .link from stage 2), that
       link is adopted: item.link = neighbour.link. The result is a direct link
       between two alert_chosen items. Meaning: choosing one of them blocks their
       shared alert_blocked neighbour, which by construction would immediately create
       an alert at the *other* linked item - so linked alert_chosen items must
       eventually be chosen together.

    4. Invariant check: a fully-blocked 2x2 block must never occur - stage 2/3's
       alert_blocked/alert_chosen/link bookkeeping exists specifically to prevent it.
       If it happens anyway, raise InvalidTilingError: that signals a bug upstream,
       not a recoverable case.

    5. assign_graph_id: give every alert_chosen item a graph_id, and reconcile it with
       its linked item's graph_id (see SquareItem.graph_id and assign_graph_id) so
       that a whole chain of mutually-linked alert_chosen items ends up sharing one
       id. The graph_id is what mark_graph_conflicts and later steps use to decide
       "if one of these gets chosen, all the others linked to it must be chosen too".

    -- Open question: does this actually bound how far a graph can spread? --
    A chain of linked alert_chosen items (stage 5) is, in principle, a coordination
    requirement that reaches across tile boundaries - exactly what the sublattice
    scheme above is designed to avoid. If these graphs could grow without bound, that
    would defeat the whole point of the parallel/sequential split (parallel
    place_square_in_core sweeps, sequential find_graphs passes): resolving one graph
    could end up requiring information from far outside the tile that started it,
    destroying parallelism.

    The working hypothesis is that running find_graphs after every single sublattice
    placement - i.e. resolving alerts immediately rather than letting them accumulate
    across passes - keeps every graph's spatial extension bounded by a small constant
    (something like ten cells), regardless of how large the overall plane is. That is
    only a hypothesis right now, not something derived above. test_graph_spread (with
    eval_graphs, still unimplemented) is meant to measure it empirically first -
    collect the graphs that form under repeated closure and look at
    max_graph_extension / max_num_nodes - before attempting an actual proof.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            neighbours = (map_of_squares[i - 1, j], map_of_squares[i + 1, j],
                          map_of_squares[i, j - 1], map_of_squares[i, j + 1])
            if all(n.state == StateEnum.blocked for n in neighbours):
                item.state = StateEnum.chosen

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.state != StateEnum.free:
                continue
            ring = [map_of_squares[i + di, j + dj] for di, dj in RING_OFFSETS]
            set_alert_blocked(item, ring)
            if item.alert_blocked:
                set_alert_chosen(item, i, j, ring)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.alert_chosen:
                link_chosen_items(item, i, j, map_of_squares)

    # A 2x2 block of blocked items must never happen - the alert_chosen/alert_blocked
    # flags exist to prevent it. Treat its occurrence as a broken invariant.
    for i in range(rows - 1):
        for j in range(cols - 1):
            corners = (map_of_squares[i, j], map_of_squares[i, j + 1],
                       map_of_squares[i + 1, j], map_of_squares[i + 1, j + 1])
            if all(c.state == StateEnum.blocked for c in corners):
                raise InvalidTilingError(f"2x2 all-blocked block at ({i}, {j})")

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.alert_chosen:
                assign_graph_id(item, i, j, rows, map_of_squares)


def mark_graph_conflicts(map_of_squares):
    """For every alert_chosen item (graph_id assumed final - run find_graphs first),
    check the 4 diagonal neighbours it would block if chosen. A diagonal neighbour
    that is itself alert_chosen with a *different* graph_id means choosing either
    graph would block a member of the other - the two graphs exclude each other, so
    each item records the other's graph_id in .conflicts (see SquareItem.conflicts
    and alert_graphs.mark_graph_conflicts, which does the actual per-item work). A
    diagonal neighbour sharing the *same* graph_id would mean the graph blocks its own
    member, which should never happen - alert_graphs.mark_graph_conflicts raises
    InvalidTilingError if it does.
    """
    rows, cols = map_of_squares.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if item.alert_chosen:
                mark_item_graph_conflicts(item, i, j, map_of_squares)


def find_central_graph_items(map_of_squares, gen):
    """Assign centrality to alert_chosen items, one generation at a time.

    gen counts how many times this function has already been called for
    map_of_squares: call it with gen=0 first, then gen=1, 2, ... on each subsequent
    call. Requires find_graphs to have already run (.alert_chosen/.link must be set).

    The function runs over all alert_chosen item. If gen=0 and the current item has a linked item 
    with no .link of its own, the linked item is a terminal item, and we set its centrality to 0.
    Each later call looks at every alert_chosen item that has a link: if the linked
    item's centrality is gen-1, the current item's centrality becomes gen. So
    centrality ends up measuring distance, in .link hops, from a terminal item
    outward, one ring further with each generation.

    Returns found: True if this call assigned centrality to at least one item, False
    otherwise - callers can loop, incrementing gen, until found comes back False to
    know every item reachable by .link chains has been assigned a centrality.


    """
    rows, cols = map_of_squares.shape
    found = False
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            item = map_of_squares[i, j]
            if not (item.alert_chosen and item.link is not None):
                continue
            linked = map_of_squares[item.link]
            if gen == 0:
                if linked.link is None:
                    linked.centrality = 0
                    found = True
            elif linked.centrality == gen - 1:
                item.centrality = gen
                found = True
    return found

def find_cycle_graphs(map_of_squares, gen):
    


#    From here on the project is no longer aiming for a
#    parallel algorithm (for now).

def do_closure(map_of_squares):
    """Sequential clean-up step run after each parallel placement sweep.

    Thin wrapper around the two stages of closure: find_graphs discovers alert_chosen
    items, links them, and groups them into graphs (see its docstring for the full
    five-stage breakdown); mark_graph_conflicts then works out, for the now-final
    graphs, which pairs of graphs exclude each other. Both are exposed as standalone
    functions so callers - e.g. test_graph_spread - can invoke find_graphs on its own.
    """
    find_graphs(map_of_squares)
    mark_graph_conflicts(map_of_squares)
