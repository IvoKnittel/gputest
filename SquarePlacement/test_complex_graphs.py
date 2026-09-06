"""A catalogue of the algorithmically-distinct .forced_by/.forces shapes that
can arise once alert_chosen items start pointing at each other - each one
built from a real grid (map_of_squares_from_array) with a handful of squares
placed to produce that specific shape via find_alerts_set_links/do_closure,
so each shape can be examined in isolation.
"""

import numpy as np

from map_of_squares import StateEnum
from representation import (build_map_of_squares,
                             map_of_squares_from_array,
                             display_closure_step)
from closure import (find_alerts_set_links,
                      find_secondary_links,
                      assign_paths,
                      get_blocked_links,
                      do_closure,
                      forced_closure,
                      place_squares)

from test_utils import place_and_chase 


def test_line():
    """A simple chain, no branching - derived from a real grid via find_alerts_set_links,
    unlike the rest of this file's hand-built shapes.

    The six placed squares' blocked diagonal neighbours pair into three mutual
    "2x2 minus one" quadrants, each alert_blocking the other: (8,2)/(7,3),
    (6,4)/(5,5), (4,6)/(3,7). (9,1) starts the cascade via a plain diagonal
    link to (7,3), not a quadrant pairing.

    Unbranched, so get_blocked_links is empty.
    """
    m = build_map_of_squares(11, 10)
    positions = [(6, 1), (9, 4), (4, 3), (7, 6), (2, 5), (5, 8)]
    place_squares(m, positions)

    find_alerts_set_links(m)
    find_secondary_links(m)
    alert_chosen_positions_asserted = [(9, 1), (7, 3), (5, 5), (3, 7)]

    chain = [alert_chosen_positions_asserted[0]]
    while len(chain) < len(alert_chosen_positions_asserted):
        chain.append(next(iter(m[chain[-1]].forces)))
    assert chain == alert_chosen_positions_asserted

    assert not m[9, 1].alert_blocked
    for pos in alert_chosen_positions_asserted[1:]:
        assert m[pos].alert_blocked

    assign_paths(m)
    assert get_blocked_links(m) == set()

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line (simple chain): alert_blocked=blue, alert_chosen=yellow, both=green',
                          show_links=True, show_real=True, colormap=colormap)


def test_tree_fan_out():
    """One item forces several others at once - motivates set_alert_chosen_set_links
    linking every free diagonal neighbour of an alert_blocked centre, not
    just the one tied to its own quadrant.

    The four squares each alert_block one of (4,4)'s diagonal neighbours via
    a separate corner, so (4,4) links to all four despite never being
    alert_chosen itself (nothing forces it) - .forces and .alert_chosen are
    independent facts.
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
    do_closure(m, "")

    assert not m[4, 4].alert_chosen and m[4, 4].forced_by == set()
    assert m[4, 4].forces == {(2, 2), (2, 6), (6, 2), (6, 6)}
    for corner in [(2, 2), (2, 6), (6, 2), (6, 6)]:
        assert m[corner].alert_chosen and (4, 4) in m[corner].forced_by

    assert get_blocked_links(m) == set()

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'tree (fan-out)', show_links=True, show_real=True, colormap=colormap)

def test_tree_fan_in():
    """Several items all point at one node in the same generation: (3, 3) is
    the fan-in point, forced_by four separate items at once - (2, 1), (4, 1),
    (4, 3), (4, 5) - none of which needs any of the others to force it.

    Demonstrated two different ways, on two fresh copies of the same board:
    placing (6, 4) forces (3, 3) two hops away, via (4, 5); placing (2, 1)
    forces it directly, one hop away. Either seed alone is enough - that's
    what makes (3, 3) a fan-in, not just a link in a chain.
    """
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
    do_closure(m,'')
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line into cycle', show_links=True, show_real=True, colormap=colormap)

    assert m[3, 3].alert_chosen and not m[3, 3].forces
    assert {(2, 1), (4, 1), (4, 3), (4, 5)} <= m[3, 3].forced_by

    place_squares(m, list(forced_closure(m, (6, 4))))
    assert m[3, 3].state == StateEnum.chosen
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'fan-in forced via (6, 4)', show_links=True, show_real=True, colormap=colormap)

    m = map_of_squares_from_array(grid)
    do_closure(m, '')
    place_squares(m, list(forced_closure(m, (2, 1))))
    assert m[3, 3].state == StateEnum.chosen
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'fan-in forced via (2, 1)', show_links=True, show_real=True, colormap=colormap)

def test_cycle_unidirectional_bidirectional():
    """ build a free cell with 4 direct blocked neighbors. """
    m = build_map_of_squares(12, 12)
    place_and_chase(m, (2, 2), "round 1: (2,2) placed", True)
    place_and_chase(m, (3, 5), "round 2: (3,5) placed", True)
    place_and_chase(m, (5, 1), "round 3: (5,1) placed", True) 
    place_and_chase(m, (6, 4), "round 4: (6,4) placed", True)   

def test_line_into_eye():
    """(1, 4) is a pure diagonal linker into (1, 6) - like (4, 4) in
    test_tree_fan_out, it has real forces without being alert_chosen itself,
    since nothing ever flags it as anyone's own corner. (1, 6) and (1, 8), at
    the far end, are a genuine mutual pair - each is alert_chosen and forces
    the other - the 2-cycle "into" which this line feeds.

    Demonstrated by actually placing (1, 4): both cycle cells end up chosen
    too, forced_closure's own chain reaching all the way round the pair, and
    nothing beyond those three cells does.
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
    do_closure(m, "")

    assert not m[1, 4].alert_chosen
    assert m[1, 6].alert_chosen
    assert m[1, 8].alert_chosen
    assert m[1, 4].forced_by == set()
    assert m[1, 4].forces == {(1,6)}
    assert m[1, 6].forces == {(1,8)}
    assert (1, 6) in m[1, 8].forced_by
    # (1, 4) has exactly one .forces target, (1, 6) - assign_paths prunes a
    # single-target entry like this rather than seeding it directly: (1, 4)
    # has no influence on which cells must be chosen together beyond "if
    # (1, 4) is chosen, (1, 6) is chosen", so (1, 6) gets treated like the
    # entry instead, and (1, 4) never receives an id of its own. Not about
    # .alert_chosen - (1, 4) isn't one, but that's not why it stays empty.
    assert m[1, 4].path_id == set()

    rows, cols = m.shape
    chosen_before = {(i, j) for i in range(rows) for j in range(cols)
                      if m[i, j].state == StateEnum.chosen}
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line into cycle: before placing (1, 4)',
                          show_links=True, show_real=True, colormap=colormap)

    place_squares(m, list(forced_closure(m, (1, 4))))

    chosen_after = {(i, j) for i in range(rows) for j in range(cols)
                     if m[i, j].state == StateEnum.chosen}
    # Placing (1, 4) forces both cycle cells chosen too, via the mutual pair
    # itself - (1, 6) and (1, 8) - and nothing beyond those three.
    assert chosen_after - chosen_before == {(1, 4), (1, 6), (1, 8)}

    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'line into cycle: after placing (1, 4)',
                          show_links=True, show_real=True, colormap=colormap)


def test_eye_outwards():
    
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)
    do_closure(m, "")
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'cycle line outwards: before placing (1, 2)',
                          show_links=True, show_real=True, colormap=colormap)

    place_squares(m, list(forced_closure(m, (1, 2))))
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, 'cycle line outwards: after placing (1, 2)',
                          show_links=True, show_real=True, colormap=colormap)
