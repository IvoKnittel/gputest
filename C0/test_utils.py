import numpy as np

from representation import display_closure_step
from closure import redo_closure, reset_alert_bookkeeping, forced_closure, place_squares

def place_and_chase(m, pos, title, show=True, show_roots_terminals=False):
    forced = forced_closure(m, pos)
    place_squares(m, list(forced))
    reset_alert_bookkeeping(m)
    redo_closure(m, title, show=False)
    if show:
        colormap = np.zeros((*m.shape, 3))
        display_closure_step(m, title, show_links=True, show_real=True, colormap=colormap,
                              show_roots_terminals=show_roots_terminals)