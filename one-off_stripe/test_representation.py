import numpy as np
import matplotlib.pyplot as plt

from representation import (map_of_squares_from_array,
                             display_map_of_squares_3States,
                             real_space_map,
                             colorize,
                             colorize_with_alerts,
                             FREE_COLOR,
                             CHOSEN_COLOR,
                             BLOCKED_COLOR,
                             MARGIN_COLOR)
from closure import (promote_isolated_free_cells,
                      find_alerts,
                      link_patches,
                      check_tiling_invariant)


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


def test_map_input_display():
    grid = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)

    fig, (ax_real, ax_map) = plt.subplots(1, 2, figsize=(10, 5))

    real_display = real_space_map(m)
    ax_real.imshow(colorize(real_display, {0: FREE_COLOR, 1: CHOSEN_COLOR}))
    ax_real.set_title('real space')
    ax_real.axis('on')
    grid_on(ax_real, *real_display.shape)

    # map_of_squares is isomorphic to the grid-intersection points of real_space:
    # the centre of square (i, j) is exactly the corner shared by the four
    # real-space cells (i,j), (i,j+1), (i+1,j), (i+1,j+1) that square occupies -
    # i.e. real-space coordinate (j + 0.5, i + 0.5). Shifting this image's extent
    # by +0.5 puts every cell centre there, and matching this axes' limits to
    # ax_real's then draws it inset within the very same frame - one cell-width
    # smaller and centred on all sides, since map_of_squares has one fewer row and
    # column than real_space.
    map_rows, map_cols = m.shape
    map_display = display_map_of_squares_3States(m)
    ax_map.set_facecolor(MARGIN_COLOR)
    ax_map.imshow(colorize(map_display, {0: FREE_COLOR, 1: CHOSEN_COLOR, -1: BLOCKED_COLOR}),
                  extent=(0, map_cols, map_rows, 0))
    ax_map.set_title('map_of_squares')
    ax_map.axis('on')
    grid_on(ax_map, map_rows, map_cols, offset=0)
    ax_map.set_xlim(ax_real.get_xlim())
    ax_map.set_ylim(ax_real.get_ylim())

    plt.tight_layout()
    plt.show()


def display_closure_step(m, title):
    """Show a single map_of_squares panel coloured via colorize_with_alerts, so
    alert_blocked (blue), alert_chosen (yellow), and both-at-once (green) are
    visible on top of the plain free/chosen/blocked colours - see
    test_do_closure_steps, which calls this after each do_closure stage in turn.
    """
    rows, cols = m.shape
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor(MARGIN_COLOR)
    ax.imshow(colorize_with_alerts(m))
    ax.set_title(title)
    ax.axis('on')
    grid_on(ax, rows, cols)
    plt.tight_layout()
    plt.show()


def test_do_closure_steps():
    """Run do_closure's stages one at a time on the directed-graph review grid from
    test_map_input_display, displaying the map after each so alert_blocked/
    alert_chosen items (and their overlap) are visible as they appear.
    """
    grid = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    display_closure_step(m, '0: initial state')

    promote_isolated_free_cells(m)
    display_closure_step(m, '1: promote_isolated_free_cells')

    find_alerts(m)
    display_closure_step(m, '2: find_alerts  (alert_blocked=blue, alert_chosen=yellow, both=green)')

    link_patches(m)
    display_closure_step(m, '3: link_patches')

    check_tiling_invariant(m)
    display_closure_step(m, '4: check_tiling_invariant  (invariant held - no state change)')
