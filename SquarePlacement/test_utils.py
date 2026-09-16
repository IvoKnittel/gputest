import json
import os
import re
from enum import Enum

from representation import RealSpaceMargin


class DoClosureSteps(Enum):
    """One entry per stage do_closure_intern's own pipeline runs, in the
    order it runs them (see its own docstring) - the vocabulary a caller's
    `asserts` argument to do_closure uses to say which intermediate state it
    wants to check, since do_closure itself now owns display/orchestration
    end to end and a caller can no longer hand-run these stages itself just
    to get an assertion point in between two of them.
    """
    find_alerts_set_links = 0
    find_secondary_links = 1
    assign_paths = 2
    get_blocked_links = 3
    dissolve_blocked_paths = 4
    place_square_in_seat_closed = 5


class DoClosureAssertsSingle:
    """One do_closure_intern pass's worth of assertion hooks. .asserts is a
    list of (DoClosureSteps, fn) pairs - fn taking the map (m) alone - built
    up by the caller (e.g. `single.asserts.append((DoClosureSteps.assign_paths,
    my_check))`) before being handed to DoClosureAsserts.first_pass or
    .second_pass. Each pipeline function in closure.py that takes its own
    `asserts` argument (find_alerts_set_links, find_secondary_links,
    assign_paths, get_blocked_links, dissolve_blocked_paths,
    place_square_in_seat_closed) looks up its own DoClosureSteps entry in
    this list once its own work is done, and calls that entry's fn(m) - see
    each function's own docstring.
    """
    def __init__(self):
        self.asserts = []


class DoClosureAsserts:
    """Assertion hooks for both of do_closure_intern's own passes (see its
    own docstring for what "first" and "second" pass mean) - .first_pass and
    .second_pass, each later assigned a DoClosureAssertsSingle. Passed as
    do_closure's own `asserts` argument; None (the default, on either field
    or on the whole object) means no assertion hooks for that pass.
    """
    def __init__(self):
        self.first_pass = None
        self.second_pass = None


class DoClosureDisplayItem:
    """One registered display_closure_step call. step: a DoClosureSteps
    member naming which pipeline stage this fires after, or None for
    do_closure_intern's own fixed end-of-pass call ("after find_secondary_links
    2nd" today - the one call in do_closure_intern that isn't gated on any
    stage's own return value) or for one of the two "very end" panels
    (check_tiling_invariant's own failure panel, or do_closure's own summary/
    on-error panel - see DoClosureDisplay.top_level). kwargs: dict forwarded
    to display_closure_step as its own keyword arguments (m and colormap are
    supplied by the caller of closure.display()/do_closure, not here -
    colormap is always a fresh np.zeros((*m.shape, 3)) per call, matching
    every existing call site). on_error: bool, default False - see
    closure.display()'s own docstring for what this flag does.
    """
    def __init__(self, step, kwargs, on_error=False):
        self.step = step
        self.kwargs = kwargs
        self.on_error = on_error


class DoClosureDisplaySingle:
    """One do_closure_intern pass's worth of display hooks - .items is a list
    of DoClosureDisplayItem, built up by the caller the same way
    DoClosureAssertsSingle.asserts is. closure.display() matches items by
    their .step (and by whether their .on_error matches the call's own
    on_error argument).
    """
    def __init__(self):
        self.items = []


class DoClosureDisplay:
    """Display hooks for do_closure_intern's own two passes - .first_pass /
    .second_pass, each None or a DoClosureDisplaySingle, matching
    DoClosureAsserts's own shape exactly; forwarded to do_closure_intern's
    own `display` argument the same way DoClosureAsserts is.

    .top_level is the one addition beyond that mirror: do_closure's own
    summary display (on success) and its own on-error summary panel both
    fire *after* do_closure_intern has already returned or raised - outside
    either of do_closure_intern's own two passes, using data (pos,
    newly_blocked, newly_chosen) only do_closure itself has. do_closure_intern
    never reads this field; do_closure reads it directly (see do_closure's
    own docstring for how). Kept on this same object anyway, rather than a
    separate parameter, so a caller only ever has to build and pass one
    display object per do_closure call - see default_display, which builds
    all three fields at once.
    """
    def __init__(self):
        self.first_pass = None
        self.second_pass = None
        self.top_level = None


def default_display(show=True, show_all=False, margin=None, roi_margin=0, title=""):
    """Builds a DoClosureDisplay reproducing exactly what do_closure/
    do_closure_intern used to show for a given (show, show_all, margin,
    roi_margin) combination, pre-display-mechanism - the drop-in replacement
    for a do_closure call site that used to just pass those flags directly.

    show_all reproduces do_closure_intern's old `show`-gated per-stage
    panels (after assign_paths/dissolve_blocked_paths/
    place_square_in_seat_closed - each still independently gated by its own
    stage's own return value, computed by do_closure_intern itself, not by
    anything registered here). The "after find_secondary_links 2nd" panel is
    registered unconditionally, regardless of show_all - preserving a
    pre-existing quirk of the original code (see do_closure_intern's own
    docstring): that one call never was gated on `show` to begin with.

    show reproduces do_closure's own old `show`-gated final summary panel
    (do_closure.top_level, on_error=False).

    Both the check_tiling_invariant failure panel (do_closure_intern's
    second_pass, on_error=True) and do_closure's own on-error summary panel
    (top_level, on_error=True) are registered regardless of show/show_all -
    matching show_on_error's own old default of True, independent of show;
    do_closure's own show_on_error argument remains the master switch for
    whether the replay that actually surfaces either of them ever runs.

    title, if given, should match the title passed to the same do_closure
    call - baked into the check_tiling_invariant failure panel's own title
    (the one display() call whose kwargs can't be patched up with a
    just-caught exception's title at call time, unlike do_closure's own
    top_level items - see do_closure's own handling of those).
    """
    display_obj = DoClosureDisplay()
    first_pass = DoClosureDisplaySingle()
    second_pass = DoClosureDisplaySingle()
    top_level = DoClosureDisplaySingle()

    common = {'show_links': True, 'show_real': True, 'margin': margin, 'roi_margin': roi_margin}

    if show_all:
        first_pass.items.append(DoClosureDisplayItem(
            DoClosureSteps.assign_paths, {**common, 'title': 'after assign_paths'}))
        first_pass.items.append(DoClosureDisplayItem(
            DoClosureSteps.dissolve_blocked_paths, {**common, 'title': 'after dissolve_blocked_paths'}))
        first_pass.items.append(DoClosureDisplayItem(
            DoClosureSteps.place_square_in_seat_closed, {**common, 'title': 'after place_square_in_seat_closed'}))

    second_pass.items.append(DoClosureDisplayItem(
        DoClosureSteps.find_secondary_links, {**common, 'title': 'after find_secondary_links 2nd'}))
    second_pass.items.append(DoClosureDisplayItem(
        None, {**common, 'title': f"ERROR: {title}: check_tiling_invariant failed", 'title_color': 'red'},
        on_error=True))

    if show:
        top_level.items.append(DoClosureDisplayItem(None, dict(common)))
    top_level.items.append(DoClosureDisplayItem(None, {**common, 'title_color': 'red'}, on_error=True))

    display_obj.first_pass = first_pass
    display_obj.second_pass = second_pass
    display_obj.top_level = top_level
    return display_obj

# Display settings for any board built (build_margin_free_map) or seeded
# (closure.add_margin_ring) with that same two-ring margin convention - an
# outermost chosen ring plus the blocked ring just inside it (see either
# function's own docstring). ROI_MARGIN=1 crops the outermost chosen ring (a
# pure construction detail) off before display; MARGIN's width=2 accounts for
# it plus the blocked ring when colouring/trimming the real-space panel.
ROI_MARGIN = 1
MARGIN = RealSpaceMargin(width=2)

_GENERATED_TEST_HEADER = '''"""Auto-generated by test_utils.write_generated_test - do not hand-edit.

Each test function below replays one scorecard's recorded call_sequence from
closure_test_scorecards.json, in order, exactly as the real do_closure (or its
hand-orchestrated equivalent - see each function's own docstring) executed it
for that test. Regenerate one test with
write_generated_test(test_id, display=...); it replaces only that test's own
BEGIN/END block below, leaving every other already-generated test untouched.
"""
import numpy as np

from map_of_squares import StateEnum, InvalidTilingError
from representation import build_map_of_squares, display_closure_step, RealSpaceMargin
from closure import (find_alerts_set_links, find_secondary_links, assign_paths,
                      get_blocked_links, dissolve_blocked_paths, place_square_in_seat_closed,
                      clear_all_but_state, check_tiling_invariant)
'''


def _render_margin(margin):
    if margin is None:
        return "None"
    return f"RealSpaceMargin(width={margin['width']}, crop={margin['crop']})"


def _render_test_body(test_id, sc, display):
    """Turns one closure_test_scorecards.json test entry into the indented
    body of a pytest function: map construction, then call_sequence replayed
    step by step (asserts as comments - their text is prose, not safe to
    exec), with each recorded display_closure_step call interleaved at its
    own position when display=True, omitted entirely when display=False.
    """
    indent = "    "
    lines = []

    lines.append(f'{indent}"""Reconstructed from closure_test_scorecards.json '
                 f'(test_id={test_id!r}).\n')
    lines.append(f"{indent}{sc['free_text']}")
    lines.append(f'{indent}"""')

    lines.append(f"{indent}m = build_map_of_squares({sc['initial_map_rows']}, "
                 f"{sc['initial_map_cols']})")
    if sc["initial_chosen_cells"]:
        lines.append(f"{indent}for pos in {sc['initial_chosen_cells']!r}:")
        lines.append(f"{indent}    m[tuple(pos)].state = StateEnum.chosen")
    if sc["initial_blocked_cells"]:
        lines.append(f"{indent}for pos in {sc['initial_blocked_cells']!r}:")
        lines.append(f"{indent}    m[tuple(pos)].state = StateEnum.blocked")
    lines.append("")

    displays_by_position = {}
    for d in sc["displays"]:
        displays_by_position.setdefault(d["position"], []).append(d)

    def emit_asserts(step_num):
        for a in sc["asserts"]:
            if a["evaluated_at_step"] == step_num:
                note = f" ({a['note']})" if a.get("note") else ""
                lines.append(f"{indent}# ASSERT: {a['assertion']}{note}")

    def emit_displays(position):
        if not display:
            return
        for d in displays_by_position.get(position, []):
            f = d["flags"]
            lines.append(f"{indent}colormap = np.zeros((*m.shape, 3))")
            lines.append(
                f"{indent}display_closure_step(m, title={test_id!r}, "
                f"show_links={f['show_links']}, show_real={f['show_real']}, "
                f"colormap=colormap, margin={_render_margin(f['margin'])}, "
                f"roi_margin={f['roi_margin']})"
            )
            if d.get("note"):
                lines.append(f"{indent}# display note: {d['note']}")

    emit_displays(0)   # position 0 = before any call_sequence step ran
    emit_asserts(0)

    for step in sc["call_sequence"]:
        fn = step["function_name"]
        n = step["step"]

        call = None
        if fn == "get_blocked_links":
            call = "p = get_blocked_links(m)"
        elif fn == "set_blocked_links":
            # closure_test_scorecards.json still names this step
            # "set_blocked_links", recorded when that was do_closure's real
            # second stage - set_blocked_links itself has since been removed
            # in favour of dissolve_blocked_paths, which do_closure actually
            # calls now, so that's what gets emitted here.
            call = "dissolve_blocked_paths(m, p)"
        elif fn == "display_closure_step":
            call = None  # regenerated via emit_displays(n) below, respecting `display`
        else:
            call = f"{fn}(m)"

        if call is not None:
            if step.get("raised"):
                lines.append(f"{indent}try:")
                lines.append(f"{indent}    {call}")
                lines.append(f"{indent}    raise AssertionError("
                             f"{fn!r} + ' was expected to raise InvalidTilingError but did not')")
                lines.append(f"{indent}except InvalidTilingError as e:")
                lines.append(f"{indent}    pass  # expected - see raised_message in the scorecard")
            else:
                lines.append(f"{indent}{call}")

        emit_asserts(n)
        emit_displays(n)

    return "\n".join(lines) + "\n"


def _block_markers(test_id):
    return (f"# === BEGIN GENERATED TEST: {test_id} ===",
            f"# === END GENERATED TEST: {test_id} ===")


def write_generated_test(test_name, display, json_path=None, output_path=None):
    """Look up test_name in closure_test_scorecards.json and (re)write a
    pytest function that replays its recorded call_sequence into
    tests_generated.py - including every recorded display_closure_step call
    (with its recorded flags) when display=True, calling only the plain
    call_sequence functions (no display) when display=False.

    Regenerating the same test_name overwrites just that one function's own
    BEGIN/END-marked block in tests_generated.py, leaving every other
    previously-generated test in the file untouched. Returns output_path.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = json_path or os.path.join(here, "closure_test_scorecards.json")
    output_path = output_path or os.path.join(here, "tests_generated.py")

    with open(json_path) as f:
        db = json.load(f)

    tests = db["tests"]
    if test_name not in tests:
        raise KeyError(f"{test_name!r} not found in {json_path} - "
                        f"available test_ids: {sorted(tests)}")

    sc = tests[test_name]
    func_name = f"{test_name}_generated"
    body = _render_test_body(test_name, sc, display)

    begin, end = _block_markers(test_name)
    block = f"{begin}\ndef {func_name}():\n{body}{end}\n"

    if os.path.exists(output_path):
        with open(output_path) as f:
            content = f.read()
    else:
        content = _GENERATED_TEST_HEADER

    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(content):
        content = pattern.sub(block.rstrip("\n"), content)
    else:
        if not content.endswith("\n\n"):
            content = content.rstrip("\n") + "\n\n\n"
        content += block

    with open(output_path, "w") as f:
        f.write(content)

    return output_path