"""Hierarchical union-find over patch ids, for the fixed-superlattice closure
refactor (see do_closure_old's own docstring for the problems motivating it).

A patch id is a cell's own (row, col) index in the base lattice. An alias
(id_a, id_b) records that two ids name the same patch; a conflict (id_a, id_b)
records that two patches are mutually exclusive. Both are stored low-id-first
so (a, b) and (b, a) are one entry in a set.

A CoreItem is one node of the pyramid: a 3x3 array of elements - base-level
squares for the lowest level, CoreItems above that - plus a margin of read-only
references around them, and the rectangular realm
((row_start, row_end), (col_start, col_end)) of base-lattice ids it owns.

Information flows both ways:
- up: .aliases and .conflicts, the pairs this core could not yet resolve
  inside its own realm (update_aliases / update_conflicts).
- down: .aliases_down and .blocked, filled by the descending pass (not built
  yet). .blocked is the set of path ids of this core that may not be placed in
  the current kernel call because they conflict with a path of another core
  placed simultaneously; it is recomputed per colorcode.
"""


class CoreItem:
    """elements: 3x3 object array (base-level squares or CoreItems, never
    mixed). margin: the read-only items around them. realm: the rectangle of
    base-lattice ids this core owns.
    """
    def __init__(self, elements, margin, realm):
        self.elements = elements
        self.margin = margin
        self.realm = realm
        self.aliases = set()
        self.conflicts = set()
        self.aliases_down = set()
        self.blocked = set()

    def update_aliases(self):
        """Recompute .aliases from the elements: every element's aliases,
        minus the pairs whose two ids both lie inside this core's realm."""
        self.aliases = self._carry_up('aliases')

    def update_conflicts(self):
        """Recompute .conflicts from the elements, same rule as
        update_aliases."""
        self.conflicts = self._carry_up('conflicts')

    def _carry_up(self, field):
        merged = set()
        for element in self.elements.flat:
            if isinstance(element, CoreItem):
                merged |= getattr(element, field)
            else:
                raise NotImplementedError(
                    f"update_{field} for base-level elements ({type(element).__name__}) "
                    f"is not defined yet")
        return {pair for pair in merged
                if not (in_realm(pair[0], self.realm) and in_realm(pair[1], self.realm))}


def in_realm(id_, realm):
    """True if id_ (a (row, col) base-lattice index) falls inside realm -
    ((row_start, row_end), (col_start, col_end)), both ends exclusive on
    row_end/col_end, matching image_to_squares.core_range_for_tile.
    """
    (row_start, row_end), (col_start, col_end) = realm
    row, col = id_
    return row_start <= row < row_end and col_start <= col < col_end
