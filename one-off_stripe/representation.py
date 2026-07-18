"""Tools for hand-building small map_of_squares scenarios and displaying their .link
structure directly, instead of only meeting chains/cycles/vertices incidentally
inside a full placement run.

A "vertex" here means a fan-in point: several items linking to the same target.
Real map_of_squares instances can mix all three shapes (linear chains, cycles, and
fan-in vertices) freely - these builders let a specific combination be stated
directly, then displayed, to explore what happens before trying to prove anything
about the general case.
"""

import numpy as np
import matplotlib.pyplot as plt

from map_of_squares import SquareItem, StateEnum, InvalidTilingError

# create maps from user input

def build_map_of_squares(rows, cols):
    """A rows x cols object array of fresh, unlinked SquareItem."""
    m = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = SquareItem()
    return m


# The four diagonal offsets place_square_in_core blocks when an item is chosen.
DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def map_of_squares_from_array(state_grid):
    """Build a map_of_squares array from a plain grid of 0/1 values (list of lists
    or a numpy array): 0 = free, 1 = chosen.

    Every diagonal neighbour of a chosen cell is set to blocked, matching the
    invariant place_square_in_core enforces elsewhere (choosing an item blocks its
    four diagonal neighbours) - an already-chosen cell is never overwritten to
    blocked, and diagonal neighbours that fall outside the grid are simply skipped,
    since this builder doesn't assume any padding margin around it.
    """
    grid = np.asarray(state_grid)
    rows, cols = grid.shape
    m = build_map_of_squares(rows, cols)

    for i in range(rows):
        for j in range(cols):
            if grid[i, j]:
                m[i, j].state = StateEnum.chosen

    for i in range(rows):
        for j in range(cols):
            if not grid[i, j]:
                continue
            for di, dj in DIAGONAL_OFFSETS:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and m[ni, nj].state == StateEnum.free:
                    m[ni, nj].state = StateEnum.blocked

    return m


def map_of_squares_to_array(map_of_squares):
    """Collapse a map_of_squares array back to a plain 0/1 grid: chosen=1, anything
    else (free or blocked) = 0.

    This is the inverse of map_of_squares_from_array: blocked cells there were
    derived from the chosen ones, not part of the original input, so they collapse
    back to 0 along with free cells - round-tripping a grid through
    map_of_squares_from_array and then this function returns the original grid.
    """
    rows, cols = map_of_squares.shape
    grid = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if map_of_squares[i, j].state == StateEnum.chosen:
                grid[i, j] = 1
    return grid

# transform

# The four diagonal offsets, checked against neighbouring block-positions rather
# than the current cell - see real_space_map.
NEIGHBOUR_DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def real_space_map(map_of_squares):
    """Convert an n x m map_of_squares into an (n+1) x (m+1) real-space grid of
    0/1 ints. A chosen item at block-position (i,j) occupies the 2x2 real-space
    block (i,j), (i,j+1), (i+1,j), (i+1,j+1), all set to 1; 0 is free.

    Two chosen squares that are direct (horizontal/vertical) neighbours in
    map_of_squares agree on a shared 1x2 or 2x1 strip of real-space cells - both
    squares stamp those same cells to 1, so that overlap is fine. Two chosen
    squares that are diagonal neighbours in map_of_squares only share a single
    real-space corner cell, which is not a valid overlap for this representation -
    raise InvalidTilingError if that is found. (Direct- and diagonal-neighbour
    overlaps are the only two overlaps geometrically possible between 2x2 squares,
    so there is no other case left to check.)
    """
    rows, cols = map_of_squares.shape
    real_space = np.zeros((rows + 1, cols + 1), dtype=int)

    for i in range(rows):
        for j in range(cols):
            if map_of_squares[i, j].state != StateEnum.chosen:
                continue

            real_space[i, j] = 1
            real_space[i, j + 1] = 1
            real_space[i + 1, j] = 1
            real_space[i + 1, j + 1] = 1

            for di, dj in NEIGHBOUR_DIAGONAL_OFFSETS:
                ni, nj = i + di, j + dj
                if (0 <= ni < rows and 0 <= nj < cols
                        and map_of_squares[ni, nj].state == StateEnum.chosen):
                    raise InvalidTilingError(
                        f"squares at ({i}, {j}) and ({ni}, {nj}) are diagonal "
                        f"neighbours - their footprints only share one real-space "
                        f"corner cell, which is not a valid overlap")

    return real_space


# display

def display_map_of_squares_3States(map_of_squares):
    """Collapse each cell's .state into a display value: chosen=1, blocked=-1, free=0."""
    rows, cols = map_of_squares.shape
    display = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            state = map_of_squares[i, j].state
            if state == StateEnum.chosen:
                display[i, j] = 1.0
            elif state == StateEnum.blocked:
                display[i, j] = -1.0
    return display


# link library


def set_link(map_of_squares, source, target):
    """Mark source as alert_chosen and link it to target; target is also marked
    alert_chosen (a linked-to item is, by construction, always alert_chosen too).
    """
    si, sj = source
    ti, tj = target
    map_of_squares[si, sj].alert_chosen = True
    map_of_squares[si, sj].link = target
    map_of_squares[ti, tj].alert_chosen = True


def build_chain(map_of_squares, cells):
    """Link cells[0] -> cells[1] -> ... -> cells[-1]. The last cell is left without
    a link, i.e. it is the chain's terminal.
    """
    for source, target in zip(cells, cells[1:]):
        set_link(map_of_squares, source, target)
    last = cells[-1]
    map_of_squares[last[0], last[1]].alert_chosen = True


def build_cycle(map_of_squares, cells):
    """Link cells[0] -> cells[1] -> ... -> cells[-1] -> cells[0]: a closed cycle
    with no terminal anywhere in it.
    """
    ring = list(cells) + [cells[0]]
    for source, target in zip(ring, ring[1:]):
        set_link(map_of_squares, source, target)


def build_vertex(map_of_squares, sources, target):
    """Link every item in sources to the same target - a fan-in/merge point."""
    for source in sources:
        set_link(map_of_squares, source, target)
    map_of_squares[target[0], target[1]].alert_chosen = True


def describe_links(map_of_squares):
    """Return one line per alert_chosen item: its index, link target (or "terminal"
    if it has none), graph_id, and centrality - the last two shown as "-" while
    still unassigned (-1), since most hand-built scenarios start out that way.
    """
    rows, cols = map_of_squares.shape
    lines = []
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.alert_chosen:
                continue
            target = str(item.link) if item.link is not None else "terminal"
            graph_id = item.graph_id if item.graph_id != -1 else "-"
            centrality = item.centrality if item.centrality != -1 else "-"
            lines.append(f"({i}, {j}) -> {target}   graph_id={graph_id} centrality={centrality}")
    return lines


def display_links(map_of_squares, title=None):
    """Plot every alert_chosen item as a dot (bigger/darker for a terminal - an
    item with no .link of its own) and draw an arrow from every linked item to its
    target, so chains, cycles, and fan-in vertices are visible at a glance.
    """
    rows, cols = map_of_squares.shape
    fig, ax = plt.subplots(figsize=(cols / 2 + 2, rows / 2 + 2))

    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.alert_chosen:
                continue
            is_terminal = item.link is None
            ax.plot(j, i, 'o',
                    markersize=14 if is_terminal else 9,
                    color='black' if is_terminal else 'tab:blue',
                    zorder=3)
            label = f"({i},{j})"
            if item.graph_id != -1:
                label += f"\ng={item.graph_id}"
            if item.centrality != -1:
                label += f" c={item.centrality}"
            ax.annotate(label, (j, i), textcoords="offset points",
                        xytext=(8, 8), fontsize=8)

    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if item.alert_chosen and item.link is not None:
                ti, tj = item.link
                ax.annotate('', xy=(tj, ti), xytext=(j, i),
                            arrowprops=dict(arrowstyle='->', color='tab:blue',
                                             shrinkA=10, shrinkB=10,
                                             connectionstyle='arc3,rad=0.15'),
                            zorder=2)

    ax.set_xlim(-1, cols)
    ax.set_ylim(rows, -1)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
