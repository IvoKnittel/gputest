import copy
import random

import numpy as np

from map_of_squares import StateEnum
from representation import map_of_squares_from_array, build_map_of_squares, display_closure_step
from closure import (find_alerts_set_links, find_secondary_links, find_secondary_links5x5,
                      clear_all_but_state, place_square)


def _forces_edges(m):
    """Every (source, target) pair m's own .forces sets encode, flattened
    into one set - the natural unit for a subset comparison between two
    find_secondary_links implementations that are no longer expected to
    agree exactly (see find_secondary_links5x5's own docstring).
    """
    rows, cols = m.shape
    return {(pos, target) for i in range(rows) for j in range(cols)
            for pos in [(i, j)] for target in m[pos].forces}


def test_window_radius_must_be_3_not_2():
    """Isolates the window-radius gap on its own, without any of the
    unrelated P=(6, 3) structure test_5x5_is_a_strict_subset_on_seat_from_
    two_alert_blocked's fixture also contains.

    Just three real blocked cells: (6, 2) and (7, 2) give P=(7, 3) its own
    alert_blocked promise (W and NW ring-adjacent, both blocked, so N=(6, 3)
    is the free third corner - the "B" this test's P forces); (4, 2) is the
    pre-existing real block the new seat find_secondary_links discovers at
    (4, 3) needs as its other three-blocked corner. Candidate a=(6, 4)'s own
    simulation chooses {a, B} together, which blocks (5, 2) (a diagonal
    neighbour of B) and (5, 3) (a diagonal neighbour of a) - completing block
    (4,2)-(5,3) into a seat at (4, 3), one row above the radius-2 window
    find_secondary_links5x5 keeps on purpose (see its own docstring) and
    find_secondary_links itself used before this exact case caught the gap.

    Confirms both sides directly: find_secondary_links (radius 3) finds the
    edge; find_secondary_links5x5 (radius 2), run on the very same board,
    does not - using the real dedicated function to demonstrate the gap
    rather than monkeypatching find_secondary_links's own window back down,
    now that a function which genuinely keeps the smaller window exists.
    """
    def build_minimal_board():
        m = build_map_of_squares(10, 8)
        for pos in [(6, 2), (7, 2), (4, 2)]:
            m[pos].state = StateEnum.blocked
        return m

    m_fixed = build_minimal_board()
    find_alerts_set_links(m_fixed)
    assert m_fixed[7, 3].alert_blocked
    find_secondary_links(m_fixed)
    assert m_fixed[6, 4].forces == {(6, 3), (4, 3)}

    m_5x5 = build_minimal_board()
    find_alerts_set_links(m_5x5)
    find_secondary_links5x5(m_5x5)
    assert m_5x5[6, 4].forces == {(6, 3)}, \
        "find_secondary_links5x5 was expected to miss (4, 3) - it didn't, so this repro no longer isolates the gap"

    # No cell starts out chosen in this minimal fixture (only the three real
    # blocks) - blocked/alert/link colouring is the whole story here, so
    # there's nothing for a red frame to mark.
    colormap = np.zeros((*m_fixed.shape, 3))
    display_closure_step(m_fixed, title="find_secondary_links (radius=3): final map",
                          show_links=True, colormap=colormap)


def test_5x5_is_a_strict_subset_on_seat_from_two_alert_blocked():
    """test_sudden_appearance.test_seat_from_two_alert_blocked's own fixture
    also contains a second, overlapping alert_blocked pair the original test
    doesn't exercise: (7, 3) is alert_blocked in its own right (not just P=
    (6, 3)'s own promised corner), with a=(6, 4) as one of its candidates
    and B=(6, 3). That candidate's own hypothetical simulation resolves a
    seat at (4, 3) - found via a block whose corners sit at row offset -3
    from P=(7, 3), outside find_secondary_links5x5's own radius-2 window.

    find_secondary_links5x5 is no longer expected to match find_secondary_
    links exactly (see its own docstring) - this fixture is a known case
    where they genuinely differ, so this test locks in the specific,
    verified difference instead: find_secondary_links5x5's edges are a
    strict subset of find_secondary_links's, missing exactly (6, 4) -> (4, 3)
    (this docstring's own case) and (7, 0) -> (9, 1) (a second, unrelated
    radius-3-only edge elsewhere in the same fixture) - never a false
    positive.
    """
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m_correct = map_of_squares_from_array(grid)
    find_alerts_set_links(m_correct)
    find_secondary_links(m_correct)

    m_5x5 = copy.deepcopy(m_correct)
    clear_all_but_state(m_5x5)
    find_alerts_set_links(m_5x5)
    find_secondary_links5x5(m_5x5)

    assert m_correct[6, 4].forces == {(6, 3), (4, 3)}
    assert m_correct[7, 5].forced_by == {(5, 4)}
    assert m_correct[7, 1].forced_by == {(5, 2), (5, 0), (7, 0)}

    correct_edges = _forces_edges(m_correct)
    edges_5x5 = _forces_edges(m_5x5)
    assert edges_5x5 <= correct_edges, f"false positive(s): {edges_5x5 - correct_edges}"
    assert correct_edges - edges_5x5 == {((6, 4), (4, 3)), ((7, 0), (9, 1))}

    # find_secondary_links never places a square - the only chosen cells on
    # the final map are the two the fixture itself seeded, (5, 1) and
    # (8, 3), framed red here to mark them as given input rather than
    # anything either implementation derived.
    originally_chosen = [(i, j) for i, row in enumerate(grid) for j, v in enumerate(row) if v]
    colormap = np.zeros((*m_correct.shape, 3))
    display_closure_step(m_correct, pos=originally_chosen,
                          title="find_secondary_links: final map",
                          show_links=True, colormap=colormap)


def test_5x5_edges_are_always_a_subset_on_random_boards():
    """find_secondary_links5x5 is a deliberately cheaper, smaller-window
    comparison point now, not a second correct implementation (see its own
    docstring) - so the property worth sweeping broadly isn't "matches
    exactly" any more, it's "never wrong": every edge it finds must also be
    one find_secondary_links (the sole correct implementation) finds. This
    runs that check across enough random boards, sizes, and fill levels that
    a genuine false positive - not just an expected miss - would show up
    somewhere in the sweep.
    """
    for seed in range(20):
        rng = random.Random(seed)
        size = rng.choice([10, 14, 18])
        m = build_map_of_squares(size, size)
        placed, attempts = 0, 0
        target = rng.randint(10, size * size // 6)
        while placed < target and attempts < 2000:
            attempts += 1
            pos = (rng.randint(1, size - 2), rng.randint(1, size - 2))
            if m[pos].state != StateEnum.free:
                continue
            try:
                place_square(m, pos)
                placed += 1
            except Exception:
                pass

        clear_all_but_state(m)
        find_alerts_set_links(m)
        m_5x5 = copy.deepcopy(m)

        find_secondary_links(m)
        find_secondary_links5x5(m_5x5)

        correct_edges = _forces_edges(m)
        edges_5x5 = _forces_edges(m_5x5)
        assert edges_5x5 <= correct_edges, \
            f"seed={seed} size={size}: false positive(s) {edges_5x5 - correct_edges}"
