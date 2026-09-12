import numpy as np

from map_of_squares import StateEnum
from representation import build_margin_free_map, margin_ring_positions, display_closure_step
from closure import place_square, do_closure
from test_utils import ROI_MARGIN, MARGIN, DoClosureAsserts, DoClosureAssertsSingle, DoClosureSteps

def test_two_forced_cells_block_each_other():
    """A 10x10 free core wrapped in build_margin_free_map's own two-ring
    margin (chosen outer ring, blocked ring just inside it - see its own
    docstring), so the forces/path_id behaviour under test comes from the
    shape itself, not from free cells touching the array's own edge - the
    same reason test_margins.py wraps its own boards this way, shared here
    via build_margin_free_map instead of a hand-built border.

    Runs entirely through one do_closure call now, via its own asserts
    argument, instead of hand-running each pipeline stage: show_initial
    hooks DoClosureSteps.find_alerts_set_links (before assign_paths/
    get_blocked_links/dissolve_blocked_paths have run - the alert/link
    structure in its rawest, pre-path form), check_and_show_after hooks
    DoClosureSteps.place_square_in_seat_closed in first_pass (checks the
    settled chosen set, displays it, and captures the resulting state for
    check_redo to compare against), and check_redo hooks the same step in
    second_pass (do_closure_intern's own silent redo round - "only display
    if it actually changed anything" is check_redo's own job, not something
    do_closure's own show flag needs to special-case). check_tiling_invariant
    isn't called by hand anywhere below - do_closure_intern already calls it
    once, unconditionally, after both passes (see its own docstring).

    Every assertion below excludes build_margin_free_map's own margin ring
    (margin_ring_positions) from whatever it checks against m's full chosen
    set - that ring's cells are chosen from construction, not from anything
    this test itself is exercising.

    Same grid, same pipeline, as test_get_and_set_blocked_links_marks_blocked_tmp
    - that test owns the specific get_blocked_links/dissolve_blocked_paths
    assertions (exactly which ids/positions get flagged, why, and which one
    of them has no local corroboration at all). This test only checks the
    board that results, and the two-round display/redo behaviour around it.
    """
    m = build_margin_free_map(10)
    size = m.shape[0]
    for pos in [(5, 8), (6, 8), (9, 4), (9, 5)]:
        place_square(m, pos)

    margin_positions = set(margin_ring_positions(size, size))
    state_before_redo = None

    def show_initial(mm):
        colormap = np.zeros((*mm.shape, 3))
        display_closure_step(mm, title='initial alerts/links, before blocked-links',
                              show_links=True, show_real=True, colormap=colormap,
                              margin=MARGIN, roi_margin=ROI_MARGIN)

    def check_and_show_after(mm):
        nonlocal state_before_redo
        # The two hand-placed dominoes survive, plus five seats that newly-
        # permanent blocking completed.
        assert {(i, j) for i in range(size) for j in range(size)
                if mm[i, j].state == StateEnum.chosen} - margin_positions == {
            (5, 8), (6, 8), (9, 4), (9, 5),
            (2, 2), (2, 11), (7, 6), (11, 2), (11, 11)}
        colormap = np.zeros((*mm.shape, 3))
        display_closure_step(mm, title='after', show_links=True, show_real=True, colormap=colormap,
                              margin=MARGIN, roi_margin=ROI_MARGIN)
        state_before_redo = [[mm[i, j].state for j in range(size)] for i in range(size)]

    def check_redo(mm):
        state_after_redo = [[mm[i, j].state for j in range(size)] for i in range(size)]
        changed = state_after_redo != state_before_redo
        assert not changed
        if changed:
            colormap = np.zeros((*mm.shape, 3))
            display_closure_step(mm, title='closure redone', show_links=True, show_real=True, colormap=colormap,
                                  margin=MARGIN, roi_margin=ROI_MARGIN)

    first_pass = DoClosureAssertsSingle()
    first_pass.asserts = [(DoClosureSteps.find_alerts_set_links, show_initial),
                           (DoClosureSteps.place_square_in_seat_closed, check_and_show_after)]
    second_pass = DoClosureAssertsSingle()
    second_pass.asserts = [(DoClosureSteps.place_square_in_seat_closed, check_redo)]

    asserts = DoClosureAsserts()
    asserts.first_pass = first_pass
    asserts.second_pass = second_pass

    do_closure(m, show=False, margin=MARGIN, roi_margin=ROI_MARGIN, asserts=asserts)

def test_get_and_set_blocked_links_marks_blocked_tmp():
    """Same shape as test_two_forced_cells_block_each_other, but stopping
    right after get_blocked_links/dissolve_blocked_paths to look at what
    they actually do. Both need path_id to already be real (run after
    assign_paths, do_closure's own order) - calling get_blocked_links straight
    after find_alerts_set_links, without assign_paths in between, leaves every path_id
    empty and returns an empty set too, not because nothing is wrong but
    because there was nothing yet for it to check.

    get_blocked_links(m) returns {45, 52, 87, 89, 119, 150} - six
    self-contradicting ids, found across the whole board, not just the two
    hand-placed squares' immediate neighbourhood. Each is a cell's own
    unique_id, shared with one of its own free diagonal neighbours: e.g. 89
    is (6, 5)'s own id - one of its diagonal neighbours ((5, 6), (7, 6), or
    (7, 4), all three) carries 89 too, so choosing (6, 5) would block a
    fellow member of its own path. dissolve_blocked_paths then blocks each
    id's own origin cell (the one whose own unique_id it is) directly, and
    does nothing else:
    (3, 3), (3, 10), (6, 3), (6, 5), (8, 7), (10, 10) all end up
    StateEnum.blocked, full stop - none of their own .forces/.forced_by is
    cleared (e.g. (3, 3) still carries forces={(2, 3), (3, 2), (2, 5), (5, 2)}
    afterward, now pointing nowhere meaningful), nothing retracts them from
    any other cell's .forces/.forced_by either, and no cell's path_id
    anywhere on the board is stripped of these six ids (38 other cells still
    carry one, unchanged). All of that is left for do_closure's own second
    round to sort out instead: clear_all_but_state wipes every cell's
    .forces/.forced_by/.path_id unconditionally, and the fresh
    find_alerts_set_links/assign_paths pass that follows re-derives
    everything from the current .state alone, in which these six cells - no
    longer free - simply drop out of consideration, the same way any other
    already-blocked cell does (see dissolve_blocked_paths' own docstring for
    the fuller argument, and the empirical check backing it).

    Once place_square_in_seat_closed runs, five of these six turn out to
    also be locally confirmed - each diagonally completes one of the five
    new seats it fills. (6, 3) is the exception: no new seat borders it, so
    nothing about the resulting board locally reveals it was ever doomed -
    get_blocked_links's path_id-membership check is the only thing that
    ever catches it (see do_closure's own docstring for why that's exactly
    the point of running this check at all).

    One do_closure call now, via its own asserts argument: show_six_blocked
    hooks DoClosureSteps.dissolve_blocked_paths in first_pass - right after
    dissolve_blocked_paths writes the real StateEnum.blocked on these six
    cells, before place_square_in_seat_closed has a chance to fill whatever
    seats that newly-permanent blocking completes. do_closure_intern's own
    second, silent pass is exactly the "clear every round's worth of stale
    bookkeeping and run the real pipeline again from scratch" step this used
    to run by hand afterward - no separate call needed for it. do_closure's
    own show=True supplies the final settled-board display.
    """
    m = build_margin_free_map(10)
    for pos in [(5, 8), (6, 8), (9, 4), (9, 5)]:
        place_square(m, pos)

    def show_six_blocked(mm):
        colormap = np.zeros((*mm.shape, 3))
        display_closure_step(mm, title='get_blocked_links/dissolve_blocked_paths: six cells now blocked',
                              show_links=True, show_real=True, colormap=colormap,
                              margin=MARGIN, roi_margin=ROI_MARGIN)

    # (6, 3) is the one self-contradicting cell with no local corroboration
    # at all: every other one of the six sits diagonal to a seat that just
    # got filled, so (6, 3) is the only one that stays invisible to a purely
    # local read of the resulting board - get_blocked_links's path_id check
    # was the only thing that ever caught it.

    first_pass = DoClosureAssertsSingle()
    first_pass.asserts = [(DoClosureSteps.dissolve_blocked_paths, show_six_blocked)]
    asserts = DoClosureAsserts()
    asserts.first_pass = first_pass

    do_closure(m, title='after blocking self-contradicting cells and filling seats', show=True,
               margin=MARGIN, roi_margin=ROI_MARGIN, asserts=asserts)
