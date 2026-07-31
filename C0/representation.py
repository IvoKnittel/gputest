"""Tools for hand-building small map_of_squares scenarios and displaying their .forces
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


def place_squares(map_of_squares, positions):
    """Set every (i, j) in positions to StateEnum.chosen on map_of_squares, in
    place. positions is expected to start out entirely free - e.g. a fresh
    map from build_map_of_squares - so no diagonal-neighbour pair among them
    can already be blocked as a side effect of some earlier placement.

    Checks for the same forbidden overlap real_space_map itself rejects: two
    placed squares that are diagonal neighbours only share a single
    real-space corner cell, not a valid overlap (see real_space_map's
    docstring) - raise InvalidTilingError if any placed pair violates it, by
    simply calling real_space_map and discarding its result, rather than
    duplicating its check here.

    Then blocks every diagonal neighbour of a placed square that is still
    free, matching the invariant place_square_in_core/map_of_squares_from_array
    enforce elsewhere (choosing an item blocks its four diagonal neighbours) -
    so a placed square's blocked neighbours show up on display_map_of_squares_3States
    too, not just the chosen square itself.
    """
    for i, j in positions:
        map_of_squares[i, j].state = StateEnum.chosen
    real_space_map(map_of_squares)

    rows, cols = map_of_squares.shape
    for i, j in positions:
        for di, dj in DIAGONAL_OFFSETS:
            ni, nj = i + di, j + dj
            if (0 <= ni < rows and 0 <= nj < cols
                    and map_of_squares[ni, nj].state == StateEnum.free):
                map_of_squares[ni, nj].state = StateEnum.blocked


def place_blocked_squares(map_of_squares, positions):
    """Set every (i, j) in positions to StateEnum.blocked on map_of_squares, in
    place - for hand-building a scenario directly from blocked cells (e.g. the
    side effect two chosen squares elsewhere would have left behind), instead of
    only ever meeting blocked cells as a derived effect of place_squares/
    map_of_squares_from_array.

    Unlike place_squares, a blocked cell has no diagonal-neighbour side effect of
    its own to apply - blocking only ever radiates out from a *chosen* square,
    never from another blocked one - so this is a direct, unconditional write.
    """
    for i, j in positions:
        map_of_squares[i, j].state = StateEnum.blocked


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


# Fixed colour scheme for every display in this module: free=white, chosen=cyan,
# blocked=gray. Grid lines (drawn separately, e.g. by a grid_on helper) are black.
# margin=light grey - for the space around an axes' actual cells when its extent is
# inset within a shared, larger frame (e.g. map_of_squares next to real_space_map).
FREE_COLOR = (1, 1, 1)
CHOSEN_COLOR = (0, 1, 1)
BLOCKED_COLOR = (0.5, 0.5, 0.5)
MARGIN_COLOR = (0.9, 0.9, 0.9)

# alert_blocked/alert_chosen overlay colours - see colorize_with_alerts. A cell can
# carry either flag alone or both at once (blue, yellow, green respectively);
# neither flag falls back to its plain state colour above.
ALERT_BLOCKED_COLOR = (0, 0, 1)
ALERT_CHOSEN_COLOR = (1, 1, 0)
ALERT_BOTH_COLOR = (0, 1, 0)


def display_map_of_squares_alerts(map_of_squares):
    """Collapse each cell's .alert_blocked/.alert_chosen flags into a display value
    for colorize: both=2, alert_chosen only=1, alert_blocked only=-1, neither=0.
    Pair with {0: <cell's plain state colour>, -1: ALERT_BLOCKED_COLOR,
    1: ALERT_CHOSEN_COLOR, 2: ALERT_BOTH_COLOR} so unflagged cells still show their
    free/chosen/blocked colour underneath.
    """
    rows, cols = map_of_squares.shape
    display = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if item.alert_blocked and item.alert_chosen:
                display[i, j] = 2.0
            elif item.alert_chosen:
                display[i, j] = 1.0
            elif item.alert_blocked:
                display[i, j] = -1.0
    return display


def colorize(display, color_by_value):
    """Turn a 2D array of values into an (rows, cols, 3) RGB image for imshow, by
    looking each cell's value up in color_by_value (a {value: (r, g, b)} mapping).
    Values not present in color_by_value default to white.
    """
    rows, cols = display.shape
    rgb = np.ones((rows, cols, 3))
    for value, color in color_by_value.items():
        rgb[display == value] = color
    return rgb


def colorize_with_alerts(map_of_squares):
    """RGB image for a map_of_squares that shows free/chosen/blocked state
    (display_map_of_squares_3States's colours) everywhere, except a cell with
    alert_blocked and/or alert_chosen raised is overridden with
    ALERT_BLOCKED_COLOR / ALERT_CHOSEN_COLOR / ALERT_BOTH_COLOR instead - so
    alert flags are visible on top of, rather than instead of, the underlying
    placement state.
    """
    state_display = display_map_of_squares_3States(map_of_squares)
    rgb = colorize(state_display, {0: FREE_COLOR, 1: CHOSEN_COLOR, -1: BLOCKED_COLOR})

    alert_display = display_map_of_squares_alerts(map_of_squares)
    rgb[alert_display == -1] = ALERT_BLOCKED_COLOR
    rgb[alert_display == 1] = ALERT_CHOSEN_COLOR
    rgb[alert_display == 2] = ALERT_BOTH_COLOR
    return rgb


def grid_on(ax, rows, cols, offset=-0.5):
    """Draw a line between every cell of a rows x cols image shown via imshow on ax
    - like MATLAB's grid on, but at cell boundaries rather than through the tick
    labels, so each field is visibly boxed off from its neighbours.

    offset is where the first boundary line sits relative to the image's own index
    origin: -0.5 (the default) matches a plain, unshifted imshow; pass 0 for an
    imshow whose extent has been shifted by +0.5, as real_space_map's grid
    intersections require (see test_map_input_display).

    Drawn as explicit lines (vlines/hlines) rather than via minor-tick gridlines,
    since minor ticks that land exactly on major tick positions (as offset=0 does)
    are not reliably drawn as gridlines across matplotlib versions/backends.
    """
    x_lines = np.arange(offset, cols + offset + 1, 1)
    y_lines = np.arange(offset, rows + offset + 1, 1)
    ax.vlines(x_lines, y_lines[0], y_lines[-1], color='black', linewidth=1)
    ax.hlines(y_lines, x_lines[0], x_lines[-1], color='black', linewidth=1)


def compute_blue_arrows(m):
    """Simplified stand-in for the patch_id/max-centrality computation
    show_pivots used to depend on, which needed find_central_patch_items to
    have fully converged - something the fan-in crowding gap on
    find_cycle_patches (see its "Known gap" note) can leave stuck indefinitely
    - and which only ever drew one arrow per patch, from its single furthest
    ("pivot") item, even when several other items link directly to the same
    terminal (see test_remove_blocked_links_disagreeing_removals: (6, 4) forces
    straight to (5, 6) just as directly as the patch's actual pivot, (2, 3),
    does, but only (2, 3) got an arrow under the old computation).

    Every alert_chosen item with no .forces of its own is a terminal by
    definition (nothing further is forced) - centrality 0, assigned directly
    here, with no generational propagation needed. Every other alert_chosen
    item's own arrow is then found by walking its .forces[0] chain forward, one
    hop at a time, until it reaches such a terminal - and the arrow is drawn
    from that starting item straight to the terminal, not just from whichever
    item happens to be furthest along the chain. A chain that loops back on
    itself before reaching a terminal (a pure ring - see find_cycle_patches)
    has nothing to resolve to, so no arrow is produced for it.

    Mutates m (sets .centrality = 0 on every terminal found). Returns a list of
    (source, terminal) position pairs, one per alert_chosen item that resolves
    to a terminal - the blue arrows to draw.
    """
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if item.alert_chosen and not item.forces:
                item.centrality = 0

    arrows = []
    for i in range(rows):
        for j in range(cols):
            item = m[i, j]
            if not item.alert_chosen or not item.forces:
                continue
            pos = (i, j)
            seen = {pos}
            while m[pos].forces:
                pos = m[pos].forces[0]
                if pos in seen:
                    pos = None
                    break
                seen.add(pos)
            if pos is not None:
                arrows.append(((i, j), pos))
    return arrows


def display_closure_step(m, title, show_links=False, show_pivots=False, show_real=False, ax=None):
    """Show a single map_of_squares panel coloured via colorize_with_alerts, so
    alert_blocked (blue), alert_chosen (yellow), and both-at-once (green) are
    visible on top of the plain free/chosen/blocked colours - see
    test_representation.test_do_closure_steps, which calls this after each
    do_closure stage in turn.

    show_links=True additionally draws a black arrow into every alert_chosen
    item from each item in its .forced_by (see SquareItem.forced_by) -
    drawn from that source, not read off the target's own .forces, so the arrow
    points the way the causal flow actually runs: source is what gets chosen
    (or blocked) first, and that is what forces this item, not the other way
    around. An item can have more than one, e.g. test_4links_situation), drawn
    directly on top of the image - the same (row, col) -> (x, y) = (col, row)
    mapping imshow already uses for the image itself, so no coordinate
    translation is needed. A self-loop (a source equal to (i, j) itself) has no
    direction to draw an arrow along, so it's circled in red instead -
    resolve_chosen_link (alert_graphs.py) skips self-loop candidates when it
    can, so this should only ever show up in a hand-built scenario, not one
    produced by find_alerts/link_patches - see
    test_link_patches_relays_past_self_loop_candidate in test_representation.py.

    show_pivots=True additionally draws a blue arrow, source to terminal, for
    every (source, terminal) pair compute_blue_arrows finds - source to
    terminal, not the other way around, to match the same causal direction the
    black .forces/.forced_by arrows point in: a terminal is what every item
    upstream of it, transitively, forces. Unlike drawing only from each patch's
    single furthest ("pivot") item, this draws one arrow per item that resolves
    to a terminal at all - see compute_blue_arrows for why, and
    test_complex_blocked_links.test_remove_blocked_links_disagreeing_removals,
    where an intermediate item like (6, 4) - not just the patch's deepest item
    - gets its own arrow too.

    show_real=True additionally draws the real-space map (real_space_map) in a
    second panel to the right, so a placement's actual physical footprint is
    visible alongside its alert/link overlay - e.g. the last, most-resolved
    step of a do_closure walkthrough (see test_do_closure_steps). Draws its
    own two-panel figure regardless of ax, since a fresh pair of axes is
    needed either way - pass ax=None (the default) whenever show_real=True.

    ax=None (default): create a new figure/axes and call plt.show() once this
    panel is drawn, as a standalone display. Pass an existing ax to draw into it
    instead, without creating a figure or calling plt.show() - e.g. to place two
    of these panels side by side in one figure, see
    test_representation.test_do_closure_steps_reverse_check.
    """
    rows, cols = m.shape
    if show_real:
        fig, (ax, ax_real) = plt.subplots(1, 2, figsize=(10, 5))
        standalone = True
    else:
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor(MARGIN_COLOR)
    ax.imshow(colorize_with_alerts(m))
    ax.set_title(title)
    ax.axis('on')
    grid_on(ax, rows, cols)

    if show_links:
        for i in range(rows):
            for j in range(cols):
                item = m[i, j]
                if not (item.alert_chosen and item.forced_by):
                    continue
                for si, sj in item.forced_by:
                    if (si, sj) == (i, j):
                        ax.plot(j, i, 'o', markersize=16, markerfacecolor='none',
                                markeredgecolor='red', markeredgewidth=2, zorder=4)
                        continue
                    # Shift both endpoints sideways, perpendicular to this
                    # arrow's own direction, so several arrows converging on
                    # the same cell from different directions (e.g. multiple
                    # items linking to the same target) fan out instead of
                    # all landing on the exact same point - see
                    # test_representation.test_do_closure_steps, where (5, 2),
                    # (7, 2), (7, 3), and (7, 5) all separately link into the
                    # (6, 3)/(6, 4) pair.
                    dx, dy = j - sj, i - si
                    length = (dx ** 2 + dy ** 2) ** 0.5
                    perp_x, perp_y = (-dy / length, dx / length) if length else (0, 0)
                    offset = 0.12
                    ax.annotate('', xy=(j + perp_x * offset, i + perp_y * offset),
                                xytext=(sj + perp_x * offset, si + perp_y * offset),
                                arrowprops=dict(arrowstyle='->', color='black',
                                                 shrinkA=8, shrinkB=8,
                                                 connectionstyle='arc3,rad=0.15'),
                                zorder=3)

    if show_pivots:
        for (pi, pj), (ti, tj) in compute_blue_arrows(m):
            if (pi, pj) == (ti, tj):
                continue
            # Shift both endpoints sideways, perpendicular to the
            # source->terminal direction, so this arrow is drawn next to -
            # rather than directly on top of - a black .forced_by arrow that
            # happens to run between (or through) the same two cells.
            dx, dy = tj - pj, ti - pi
            length = (dx ** 2 + dy ** 2) ** 0.5
            perp_x, perp_y = (-dy / length, dx / length) if length else (0, 0)
            offset = 0.25
            ax.annotate('', xy=(tj + perp_x * offset, ti + perp_y * offset),
                        xytext=(pj + perp_x * offset, pi + perp_y * offset),
                        arrowprops=dict(arrowstyle='->', color='blue',
                                         linewidth=2, shrinkA=10, shrinkB=10,
                                         connectionstyle='arc3,rad=0.0'),
                        zorder=6)

    if show_real:
        real_display = real_space_map(m)
        ax_real.imshow(colorize(real_display, {0: FREE_COLOR, 1: CHOSEN_COLOR}))
        ax_real.set_title('real space')
        ax_real.axis('on')
        grid_on(ax_real, *real_display.shape)

    if standalone:
        plt.tight_layout()
        plt.show()


# link library


def set_link(map_of_squares, source, target):
    """Mark source as alert_chosen and link it to target; target is also marked
    alert_chosen (a linked-to item is, by construction, always alert_chosen too).
    target's .forced_by gets source appended too (see SquareItem.forced_by),
    the incoming-side record of this same pairing.
    """
    si, sj = source
    ti, tj = target
    map_of_squares[si, sj].alert_chosen = True
    map_of_squares[ti, tj].forced_by.append(source)
    map_of_squares[si, sj].forces.append(target)
    map_of_squares[ti, tj].alert_chosen = True


def build_cycle(map_of_squares, cells):
    """Link cells[0] -> cells[1] -> ... -> cells[-1] -> cells[0]: a closed cycle
    with no terminal anywhere in it.
    """
    ring = list(cells) + [cells[0]]
    for source, target in zip(ring, ring[1:]):
        set_link(map_of_squares, source, target)


def describe_links(map_of_squares):
    """Return one line per alert_chosen item: its index, link target(s) (or
    "terminal" if it has none), patch_id, and centrality - the last two shown as
    "-" while still unassigned (-1), since most hand-built scenarios start out
    that way. An item linked to more than one target (see SquareItem.forces)
    lists all of them, comma-separated.
    """
    rows, cols = map_of_squares.shape
    lines = []
    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.alert_chosen:
                continue
            target = ", ".join(str(t) for t in item.forces) if item.forces else "terminal"
            patch_id = item.patch_id if item.patch_id != -1 else "-"
            centrality = item.centrality if item.centrality != -1 else "-"
            lines.append(f"({i}, {j}) -> {target}   patch_id={patch_id} centrality={centrality}")
    return lines


def display_links(map_of_squares, title=None, ax=None):
    """Plot every alert_chosen item as a dot (bigger/darker for a terminal - an
    item with no .forces of its own) and draw an arrow from every linked item to
    each of its targets (see SquareItem.forces), so chains, cycles, fan-in
    vertices, and an item linked to several targets at once are all visible at a
    glance.

    ax=None (default): create a new figure/axes and call plt.show() once this
    panel is drawn, as a standalone display - matching display_closure_step's
    ax=None convention. Pass an existing ax to draw into it instead, without
    creating a figure or calling plt.show() - e.g. to place this alongside
    other panels in one figure, see display_graph_map_and_real_space.
    """
    rows, cols = map_of_squares.shape
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(cols / 2 + 2, rows / 2 + 2))

    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.alert_chosen:
                continue
            is_terminal = not item.forces
            ax.plot(j, i, 'o',
                    markersize=14 if is_terminal else 9,
                    color='black' if is_terminal else 'tab:blue',
                    zorder=3)
            label = f"({i},{j})"
            if item.patch_id != -1:
                label += f"\np={item.patch_id}"
            if item.centrality != -1:
                label += f" c={item.centrality}"
            ax.annotate(label, (j, i), textcoords="offset points",
                        xytext=(8, 8), fontsize=8)

    for i in range(rows):
        for j in range(cols):
            item = map_of_squares[i, j]
            if not item.alert_chosen:
                continue
            for ti, tj in item.forces:
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
    if standalone:
        plt.tight_layout()
        plt.show()


def display_real_space_and_map(map_of_squares, ax_real, ax_map):
    """Draw the real-space grid and the map_of_squares grid into the given axes,
    aligned so map_of_squares sits inset within the same frame as real_space:
    map_of_squares is isomorphic to the grid-intersection points of real_space
    (the centre of square (i, j) is exactly the corner shared by the four
    real-space cells (i,j), (i,j+1), (i+1,j), (i+1,j+1) that square occupies -
    i.e. real-space coordinate (j + 0.5, i + 0.5)), so shifting this image's
    extent by +0.5 and matching ax_map's limits to ax_real's draws it inset
    within the very same frame - one cell-width smaller and centred on all
    sides, since map_of_squares has one fewer row and column than real_space.

    Shared by test_map_input_display/test_4links_situation and by
    display_graph_map_and_real_space's own "real space"/"map_of_squares" panels.
    """
    real_display = real_space_map(map_of_squares)
    ax_real.imshow(colorize(real_display, {0: FREE_COLOR, 1: CHOSEN_COLOR}))
    ax_real.set_title('real space')
    ax_real.axis('on')
    grid_on(ax_real, *real_display.shape)

    map_rows, map_cols = map_of_squares.shape
    map_display = display_map_of_squares_3States(map_of_squares)
    ax_map.set_facecolor(MARGIN_COLOR)
    ax_map.imshow(colorize(map_display, {0: FREE_COLOR, 1: CHOSEN_COLOR, -1: BLOCKED_COLOR}),
                  extent=(0, map_cols, map_rows, 0))
    ax_map.set_title('map_of_squares')
    ax_map.axis('on')
    grid_on(ax_map, map_rows, map_cols, offset=0)
    ax_map.set_xlim(ax_real.get_xlim())
    ax_map.set_ylim(ax_real.get_ylim())


def display_graph_map_and_real_space(map_of_squares, title=None):
    """Show three panels side by side (landscape): the .forced_by/.forces
    graph (display_links), the map_of_squares grid, and the real-space grid -
    so a hand-built scenario's link structure and its (if any) placement state
    are both visible at once, in one figure. See test_complex_graphs.py, whose
    scenarios only ever set .alert_chosen/.forced_by/.forces and never
    .state, so the latter two panels show an all-free grid there - the point
    is still to have all three views in one consistent, comparable layout.
    """
    fig, (ax_graph, ax_map, ax_real) = plt.subplots(1, 3, figsize=(15, 5))
    display_links(map_of_squares, title='graph', ax=ax_graph)
    display_real_space_and_map(map_of_squares, ax_real, ax_map)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
