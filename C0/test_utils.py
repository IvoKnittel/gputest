from representation import place_squares, display_closure_step
from closure import find_alerts, link_patches, remove_blocked_links, forced_closure, reset_alert_bookkeeping, fill_isolated_free_cells_once

def place_and_chase(m, pos, title, show=True):
    forced = forced_closure(m, pos)
    place_squares(m, list(forced))
    reset_alert_bookkeeping(m)
    find_alerts(m); link_patches(m); remove_blocked_links(m)
    fill_isolated_free_cells_once(m)

    if show:
        display_closure_step(m, title, show_links=True, show_real=True)

    fill_isolated_free_cells_once(m)

    if show:
        display_closure_step(m, title, show_links=True, show_real=True)

