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
from closure import get_blocked_links, do_closure, place_square

from test_utils import DoClosureAsserts, DoClosureAssertsSingle, DoClosureSteps, default_display


def test_line():
    """A simple chain, no branching - derived from a real grid via do_closure's
    own find_alerts_set_links/find_secondary_links stages, unlike the rest of
    this file's hand-built shapes.

    The six placed squares' blocked diagonal neighbours pair into three mutual
    "2x2 minus one" quadrants, each alert_blocking the other: (8,2)/(7,3),
    (6,4)/(5,5), (4,6)/(3,7). (9,1) starts the cascade via a plain diagonal
    link to (7,3), not a quadrant pairing.

    Unbranched, so get_blocked_links is empty.

    Checked via do_closure's own asserts argument instead of hand-running the
    pipeline stage by stage: check_chain hooks DoClosureSteps.find_secondary_
    links (the chain/.alert_blocked bookkeeping is only fully formed once
    both find_alerts_set_links and find_secondary_links have run, so it hangs
    off the second of the two), and check_no_blocked_links hooks
    DoClosureSteps.get_blocked_links. Both are only meaningful in do_closure's
    first, displayed pass - the shape they check never survives round 2's
    clear_all_but_state/re-derivation unchanged in a way that would need
    checking twice - so both live in first_pass only, not second_pass.

    
    default_display(show_all=True), not plain default_display(): nothing here
    ever changes .state (no seat gets filled, get_blocked_links is empty),
    only bookkeeping (.forces/.alert_chosen) - do_closure's own final summary
    display only fires on a .state change (see its own docstring), so plain
    default_display() (show=True, show_all=False) would silently show
    nothing at all. show_all=True instead registers do_closure_intern's own
    "after assign_paths" item, which fires whenever has_alert_bookkeeping(m)
    is true (regardless of .state) - exactly this shape - which is what
    actually shows the alert_blocked=blue/alert_chosen=yellow/both=green
    overlay this docstring's own title describes (that display's own title is
    the fixed "after assign_paths", not this call's title - the title above
    only labels do_closure's own, here-unused, final summary).
    """
    m = build_map_of_squares(11, 10)
    positions = [(6, 1), (9, 4), (4, 3), (7, 6), (2, 5), (5, 8)]
    for pos in positions:
        place_square(m, pos)

    def check_chain(mm):
        alert_chosen_positions_asserted = [(9, 1), (7, 3), (5, 5), (3, 7)]

        chain = [alert_chosen_positions_asserted[0]]
        while len(chain) < len(alert_chosen_positions_asserted):
            chain.append(next(iter(mm[chain[-1]].forces)))
        assert chain == alert_chosen_positions_asserted

        assert not mm[9, 1].alert_blocked
        for pos in alert_chosen_positions_asserted[1:]:
            assert mm[pos].alert_blocked

    def check_no_blocked_links(mm):
        assert get_blocked_links(mm) == set()

    first_pass = DoClosureAssertsSingle()
    first_pass.asserts = [(DoClosureSteps.find_secondary_links, check_chain),
                           (DoClosureSteps.get_blocked_links, check_no_blocked_links)]
    asserts = DoClosureAsserts()
    asserts.first_pass = first_pass

    do_closure(m, title='line (simple chain): alert_blocked=blue, alert_chosen=yellow, both=green',
               display=default_display(show_all=True), asserts=asserts)


def test_tree_fan_out():
    """One item forces several others at once - motivates set_alert_chosen_set_links
    linking every free diagonal neighbour of an alert_blocked centre, not
    just the one tied to its own quadrant.

    The four squares each alert_block one of (4,4)'s diagonal neighbours via
    a separate corner, so (4,4) links to all four despite never being
    alert_chosen itself (nothing forces it) - .forces and .alert_chosen are
    independent facts.

    Checked via do_closure's own asserts argument: check_fan_out hooks
    DoClosureSteps.find_alerts_set_links directly (this shape comes from
    set_alert_chosen_set_links alone, no find_secondary_links relay
    involved), check_no_blocked_links hooks DoClosureSteps.get_blocked_links
    - both first_pass only, same reasoning as test_line. default_display(
    show_all=True) for the same reason as test_line too: nothing here changes
    .state (no seat forms, get_blocked_links is empty), so plain
    default_display() would display nothing - show_all=True's "after
    assign_paths" item (gated on has_alert_bookkeeping, true here) is what
    actually shows this shape.
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

    def check_fan_out(mm):
        assert not mm[4, 4].alert_chosen and mm[4, 4].forced_by == set()
        assert mm[4, 4].forces == {(2, 2), (2, 6), (6, 2), (6, 6)}
        for corner in [(2, 2), (2, 6), (6, 2), (6, 6)]:
            assert mm[corner].alert_chosen and (4, 4) in mm[corner].forced_by

    def check_no_blocked_links(mm):
        assert get_blocked_links(mm) == set()

    first_pass = DoClosureAssertsSingle()
    first_pass.asserts = [(DoClosureSteps.find_alerts_set_links, check_fan_out),
                           (DoClosureSteps.get_blocked_links, check_no_blocked_links)]
    asserts = DoClosureAsserts()
    asserts.first_pass = first_pass

    do_closure(m, title='tree (fan-out)', display=default_display(show_all=True), asserts=asserts)

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
    do_closure(m, display=default_display())
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='line into cycle', show_links=True, show_real=True, colormap=colormap)

    assert m[3, 3].alert_chosen and not m[3, 3].forces
    assert {(2, 1), (4, 1), (4, 3), (4, 5)} <= m[3, 3].forced_by

    do_closure(m, (6, 4), display=default_display(show=False))
    assert m[3, 3].state == StateEnum.chosen
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='fan-in forced via (6, 4)', show_links=True, show_real=True, colormap=colormap)

    m = map_of_squares_from_array(grid)
    do_closure(m, display=default_display())
    do_closure(m, (2, 1), display=default_display(show=False))
    assert m[3, 3].state == StateEnum.chosen
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='fan-in forced via (2, 1)', show_links=True, show_real=True, colormap=colormap)

def test_cycle_unidirectional_bidirectional():
    """ build a free cell with 4 direct blocked neighbors. """
    m = build_map_of_squares(12, 12)
    do_closure(m, (2, 2), "round 1: (2,2) placed", display=default_display(show=True))
    do_closure(m, (3, 5), "round 2: (3,5) placed", display=default_display(show=True))
    do_closure(m, (5, 1), "round 3: (5,1) placed", display=default_display(show=True))
    do_closure(m, (6, 4), "round 4: (6,4) placed", display=default_display(show=True))

def test_line_into_eye():
    """(1, 4) is a pure diagonal linker into (1, 6) - like (4, 4) in
    test_tree_fan_out, it has real forces without being alert_chosen itself,
    since nothing ever flags it as anyone's own corner. (1, 6) and (1, 8), at
    the far end, are a genuine mutual pair - each is alert_chosen and forces
    the other - the 2-cycle "into" which this line feeds.

    Demonstrated by actually placing (1, 4): both cycle cells end up chosen
    too, do_closure's own chase reaching all the way round the pair, and
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
    do_closure(m, display=default_display())

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
    display_closure_step(m, title='line into cycle: before placing (1, 4)',
                          show_links=True, show_real=True, colormap=colormap)

    do_closure(m, (1, 4), 'line into cycle: after placing (1, 4)', display=default_display())

    chosen_after = {(i, j) for i in range(rows) for j in range(cols)
                     if m[i, j].state == StateEnum.chosen}
    # Placing (1, 4) forces both cycle cells chosen too, via the mutual pair
    # itself - (1, 6) and (1, 8) - and nothing beyond those three.
    assert chosen_after - chosen_before == {(1, 4), (1, 6), (1, 8)}

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
    do_closure(m, display=default_display())
    colormap = np.zeros((*m.shape, 3))
    display_closure_step(m, title='cycle line outwards: before placing (1, 2)',
                          show_links=True, show_real=True, colormap=colormap)

    do_closure(m, (1, 2), 'cycle line outwards: after placing (1, 2)', display=default_display())