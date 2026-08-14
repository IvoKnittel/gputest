"""A catalogue of the algorithmically-distinct .forced_by/.forces shapes that
can arise once alert_chosen items start pointing at each other - some derived
from a real grid via find_alerts, others hand-built directly with
the "link library" in representation.py (set_link/build_cycle), so each shape
can be examined in isolation.
"""

import numpy as np

from representation import (build_map_of_squares,
                             set_link,
                             build_cycle,
                             map_of_squares_from_array,
                             display_graph_map_and_real_space,
                             display_closure_step)
from closure import (find_alerts,
                      redo_closure,
                      place_squares)


def test_line():
    """A simple chain, no branching anywhere: every item has exactly one item
    pointing at it and points at exactly one item itself, in a straight run
    down to a single terminal - this one, unlike every other test in this
    file, is derived from a real grid via find_alerts rather
    than hand-built with the link library, so the cascade below is genuinely
    computed, not merely asserted.

    place_squares's six chosen squares each block two diagonal neighbours;
    those neighbours pair up three at a time into "2x2 minus one" quadrants:
    (7, 2)+(8, 3) around (8, 2)/(7, 3), (5, 4)+(6, 5) around (6, 4)/(5, 5),
    and (3, 6)+(4, 7) around (4, 6)/(3, 7). Each such quadrant is mutual -
    both its free corners qualify as each other's alert -
    so (8, 2), (6, 4), (4, 6) end up alert_blocked, and (7, 3), (5, 5), (3, 7)
    become "a new alert_blocked" right back, each pair completing the next.
    (9, 1) itself is not part of any such pair - it is simply diagonally
    adjacent to (8, 2), and find_alerts picks that up as a real (non-self-loop)
    link straight to (7, 3), starting the cascade without needing an
    alert_blocked of its own.

    (Tracing past (3, 7) the same way loops back round through (4, 6), (6, 4),
    (8, 2) and into (7, 3) again - the mutual pairing above means the cascade
    does not stop dead at (3, 7), it is just where this test stops looking.)
    """
    m = build_map_of_squares(11, 10)
    positions = [(6, 1), (9, 4), (4, 3), (7, 6), (2, 5), (5, 8)]
    place_squares(m, positions)

    find_alerts(m)

    alert_chosen_positions_asserted = [(9, 1), (7, 3), (5, 5), (3, 7)]

    chain = [alert_chosen_positions_asserted[0]]
    while len(chain) < len(alert_chosen_positions_asserted):
        chain.append(next(iter(m[chain[-1]].forces)))
    assert chain == alert_chosen_positions_asserted

    assert not m[9, 1].alert_blocked
    for pos in alert_chosen_positions_asserted[1:]:
        assert m[pos].alert_blocked

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line (simple chain): alert_blocked=blue, alert_chosen=yellow, both=green',
                          show_links=True, show_real=True, colormap=colormap)


def test_tree_fan_out():
    """One item points at several others at once - the "several diagonal
    neighbours qualify at once" shape that motivated set_alert_chosen linking
    every free diagonal neighbour of an alert_blocked centre, not just the one
    geometrically tied to its own quadrant.

    The four chosen squares each independently make one of (4,4)'s own four
    diagonal ring neighbours - (3,3), (3,5), (5,3), (5,5) - alert_blocked, each
    with its own separate corner: (2,2), (2,6), (6,2), (6,6) respectively.
    (4,4) itself, being free and diagonal to all four, is directly linked to
    all four corners by set_alert_chosen - not because (4,4) is itself promised
    (nothing obligates *it* to be chosen; forced_by is empty), but because
    *choosing* it would, as a side effect, block all four alert_blocked
    neighbours at once. .alert_chosen and .forces are independent facts now:
    (4,4) genuinely has real forces without being alert_chosen itself - a
    conditional "if I'm ever chosen, these four follow" available for whoever
    might place it, not a promise that it will be.
    """
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    redo_closure(m, "")

    assert not m[4, 4].alert_chosen and m[4, 4].forced_by == set()
    assert m[4, 4].forces == {(2, 2), (2, 6), (6, 2), (6, 6)}
    for corner in [(2, 2), (2, 6), (6, 2), (6, 6)]:
        assert m[corner].alert_chosen and (4, 4) in m[corner].forced_by

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'tree (fan-out)', show_links=True, show_real=True, colormap=colormap)

def test_tree_fan_in():
    """Several items all point at one node in the same generation."""
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    redo_closure(m,'')
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line into cycle', show_links=True, show_real=True, colormap=colormap)


def test_line_into_cycle():
    """(1, 4) is a pure diagonal linker into (1, 6) - like (4, 4) in
    test_tree_fan_out, it has real forces without being alert_chosen itself,
    since nothing ever flags it as anyone's own corner. (1, 6) and (1, 8), at
    the far end, are a genuine mutual pair - each is alert_chosen and forces
    the other - the 2-cycle "into" which this line feeds.
    """
    grid = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    redo_closure(m, "")

    assert not m[1, 4].alert_chosen
    assert m[1, 6].alert_chosen
    assert m[1, 8].alert_chosen
    assert m[1, 4].forced_by == set()
    assert m[1, 4].forces == {(1,6)}
    assert m[1, 6].forces == {(1,8)}
    assert (1, 6) in m[1, 8].forced_by
    # (1, 4) is a pure linker, never itself alert_chosen (see the docstring
    # above) - path_id is only ever assigned to alert_chosen items, so unlike
    # (1, 6)/(1, 8) it never gets one.
    assert m[1, 4].path_id == set()
    # (1, 6)/(1, 8) are a pure 2-cycle - no terminal exists until
    # find_cycle_patches's ring-leader election opens one, so asserting mere
    # equality here isn't enough: both would trivially equal set() if the
    # cycle never got opened at all. (1, 8) has the larger flattened index
    # (18 > 16), so it deterministically wins the election and becomes the
    # terminal.
    assert m[1, 8].centrality == 0
    assert m[1, 6].centrality == 1
    assert m[1, 6].path_id and m[1, 6].path_id == m[1, 8].path_id
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line into cycle', show_links=True, show_real=True, colormap=colormap)


def test_several_lines_into_cycle():
    """Several separate tails feeding into the same cycle at once, with only
    one of the incoming tails genuinely on the ring itself - the concrete
    shape that crowds find_cycle_patches's scalar max_id (see its "Known gap"
    docstring note).

    Never actually run through the closure pipeline: display_graph_map_and_
    real_space only draws the .forces/.forced_by graph as built here, it
    doesn't call resolve_cycles_and_centrality - so this shape's effect on
    find_cycle_patches is illustrated, not exercised or verified by this test.
    """
    m = build_map_of_squares(8, 12)
    build_cycle(m, [(3, 4), (3, 8)])
    set_link(m, (1, 2), (3, 4))
    set_link(m, (1, 6), (3, 4))
    set_link(m, (5, 4), (3, 4))
    set_link(m, (1, 10), (3, 8))
    display_graph_map_and_real_space(m, title='several lines into cycle')


def test_cycle_line_outwards():
    """A cycle where one member *also* points outward at something beyond the
    ring - the "link out of a cycle" shape find_cycle_patches's own logic
    doesn't account for: ring members and tails-feeding-in are the only two
    shapes it handles, and a node with both an in-ring link and an outward
    one at once is neither. That's a claim about the logic, not something
    this test actually exercises: display_graph_map_and_real_space only draws
    the .forces/.forced_by graph as built here, it never calls
    resolve_cycles_and_centrality, so nothing here has actually run this
    shape through find_cycle_patches to confirm what happens.
    """
    m = build_map_of_squares(8, 10)
    build_cycle(m, [(3, 4), (3, 7)])
    set_link(m, (3, 4), (6, 4))
    display_graph_map_and_real_space(m, title='cycle with a line outwards')


def test_line_inwards_cycle_line_outwards():
    """The fully general combination: one or more tails feeding into a cycle,
    and a separate line leading back out of it - everything the two simpler
    "line into cycle" and "cycle line outwards" cases show, at once. Same
    caveat as both of those: display_graph_map_and_real_space only draws the
    graph as built, it never runs resolve_cycles_and_centrality, so this
    combination's effect on find_cycle_patches is illustrated, not verified.
    """
    m = build_map_of_squares(10, 12)
    build_cycle(m, [(4, 5), (4, 8)])
    set_link(m, (1, 2), (4, 5))
    set_link(m, (2, 8), (4, 5))
    set_link(m, (4, 8), (7, 8))
    display_graph_map_and_real_space(m, title='line inwards - cycle - line outwards')
