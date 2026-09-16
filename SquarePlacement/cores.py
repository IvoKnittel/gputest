"""Hierarchical union-find over patch ids, for the fixed-superlattice closure
refactor (see do_closure_old's own docstring for the problems motivating it).

A patch id is a cell's own (row, col) index in the base lattice - "patch id =
the unique id of one of their cells", per the design this implements. An
alias (id_a, id_b) records that two ids name the same patch, without
committing to which one is canonical - picking one canonical id per patch
("giving patches their proper name") is a later, downward pass over this
same hierarchy; this module only builds the hierarchy's upward, aliasing
half.

A core covers a fixed rectangular realm of the base lattice - one core at
level 0, a 3x3 block of level-0 cores at level 1 ("core2"), and so on.
CoreItem itself carries no notion of its own realm or level - that context
is only ever meaningful relative to whichever level you're looking at it
from, so in_realm/build_parent_core take it as an explicit argument instead,
the same rectangular ((row_start, row_end), (col_start, col_end)) shape
image_to_squares.core_range_for_tile already uses.
"""


class CoreItem:
    """One core's own unresolved patch-id aliases. .aliases is a set of
    (id, id) tuples - each pair stored low-id-first (sorted) so that (a, b)
    and (b, a) are always recognised as the same alias when merged into a
    set, rather than kept as two distinct entries.
    """
    def __init__(self, aliases=None):
        self.aliases = set()
        if aliases is not None:
            for id_a, id_b in aliases:
                self.aliases.add(tuple(sorted((id_a, id_b))))


def in_realm(id_, realm):
    """True if id_ (a (row, col) base-lattice index) falls inside realm -
    ((row_start, row_end), (col_start, col_end)), both ends exclusive on
    their own row_end/col_end, matching core_range_for_tile's convention.
    """
    (row_start, row_end), (col_start, col_end) = realm
    row, col = id_
    return row_start <= row < row_end and col_start <= col < col_end


def build_parent_core(children, realm):
    """Build the CoreItem one level up from `children` (an iterable of the
    child-level CoreItems the new, bigger core covers - e.g. the 3x3 block
    of level-0 cores under one level-1 "core2"), given the new core's own
    realm.

    Merges every child's .aliases into one set, then keeps only the pairs
    still foreign to this bigger realm: an alias (id_a, id_b) is dropped -
    both ends now provably belong to one core this level already owns
    outright, so there's nothing left for any higher level to resolve -
    only when BOTH ids fall inside realm. Otherwise (at least one id still
    reaches beyond even this bigger realm) it's carried up unresolved, in
    the new core's own .aliases, for the next level up to try again. Realm
    boundaries only ever grow level over level, so an alias still foreign
    here stays a candidate to resolve later - never silently dropped.
    """
    merged = set()
    for child in children:
        merged.update(child.aliases)
    parent = CoreItem()
    for id_a, id_b in merged:
        if in_realm(id_a, realm) and in_realm(id_b, realm):
            continue
        parent.aliases.add((id_a, id_b))
    return parent
