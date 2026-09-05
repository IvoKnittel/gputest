import numpy as np

from representation import display_closure_step
from closure import do_closure, clear_all_but_state, forced_closure, place_squares

def place_and_chase(m, pos, title, show=True, show_entries_terminals=False):
    forced = forced_closure(m, pos)
    place_squares(m, list(forced))
    clear_all_but_state(m)
    do_closure(m, title, show=False)
    if show:
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap,
                              show_entries_terminals=show_entries_terminals)