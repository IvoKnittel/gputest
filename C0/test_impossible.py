"""Push the pipeline until it breaks - display only, no assertions on the happy
path. Contrast with test_simple_situations.py: that file's scenarios stay inside
what the pipeline is actually meant to handle; this one deliberately goes further,
to see what "impossible" looks like when it happens.
"""

from map_of_squares import InvalidTilingError
from representation import build_map_of_squares, place_squares, display_closure_step
from closure import (promote_isolated_free_cells,
                      find_alerts,
                      link_patches,
                      remove_blocked_links,
                      check_tiling_invariant)


def test_pinwheel():
    """Four chosen squares placed all at once, arranged in a 90-degree-rotated
    ("pinwheel") pattern around a shared centre: (3,3), (4,5), (6,4), (5,2). None
    of the four is a diagonal neighbour of another, so place_squares accepts all
    four without complaint - but their diagonal-blocking footprints close in on
    the centre from four different directions at once: exactly the "closing in
    from all four sides" scenario docs/rose_cascades_and_holes/README.md's hole
    example describes, except reached here by one direct placement instead of
    hand-forcing the blocked cells. Runs the same sequence test_rectangle does,
    in the same order, stopping at whichever stage actually raises.
    """
    m = build_map_of_squares(9, 9)
    place_squares(m, [(3, 3), (4, 5), (6, 4), (5, 2)])
    display_closure_step(m, '0: initial state', show_links=True, show_real=True)

def test_other_full_2x2():

    m = build_map_of_squares(9, 9)
    place_squares(m, [(3, 3), (4, 3), (5, 6), (6, 6)])
    display_closure_step(m, '0: initial state', show_links=True, show_real=True)
    find_alerts(m)
    link_patches(m)
    remove_blocked_links(m)
    display_closure_step(m, '0: next state', show_links=True, show_real=True)
if __name__ == "__main__":
    test_pinwheel()
