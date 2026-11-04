"""Tests for bruteforce's compound-id building blocks: create_compound_ids,
generate_compound_candidates, prune_compound_candidates,
prune_blocked_compound_candidates, compoundize_forced_by, and
propagate_compound_links. Split out of test_context_tiles.py and rewritten
to assert the full expected literal data (2026-11-01, Ivo: "make your
tests explicit and put them into a file test_compounds.py") - instead of
deriving/checking the expected result piecemeal (dict lookups, set
differences), each test below compares the function's actual return value
against a fully spelled-out literal, so the expected data is visible in the
test itself rather than computed by it.

The first four functions are not wired into anything yet - see TODO.md
item 2. compoundize_forced_by/propagate_compound_links (also item 2) are
the first step that actually writes compounds into real SquareItem
.forced_by/.forces fields, though still not called from anywhere outside
these tests.
"""
import numpy as np

from bruteforce import (create_compound_ids, generate_compound_candidates,
                         prune_compound_candidates, prune_blocked_compound_candidates,
                         compoundize_forced_by, propagate_compound_links)
from map_of_squares import SquareItem, PlacementStatus


def test_create_compound_ids_enumerates_every_subset_size_two_and_up():
    """create_compound_ids (step 2 of Ivo's own base-level ordering,
    renamed from "create_virtual_sync_id" - "virtual id" is now "compound
    id"): every non-trivial subset (size >= 2) of the given ids gets its
    own fresh compound id - 4 subsets for 3 ids (3 pairs + 1 triple), and
    a genuine size-4 subset for 4 ids ("i still maintain n-aries with n>3
    matter" - Ivo), confirming this isn't silently capped at triples.
    Compound id values are only checked for uniqueness and shape (tagged,
    distinct from a real id), never for any particular numbering -
    "arbitrary but unique" (Ivo)."""
    ids3 = ("a", "b", "c")
    result3 = create_compound_ids(ids3)
    members_only = {members for _cid, members, _is_subset_of in result3}
    assert members_only == {("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")}
    assert len(result3) == 4
    compound_id_values = [cid for cid, _members, _is_subset_of in result3]
    assert len(set(compound_id_values)) == len(compound_id_values), "every compound id must be unique"
    for cid in compound_id_values:
        assert cid not in ids3, "a compound id must never collide with one of the real ids"

    ids4 = ("a", "b", "c", "d")
    result4 = create_compound_ids(ids4)
    members_only4 = {members for _cid, members, _is_subset_of in result4}
    assert ("a", "b", "c", "d") in members_only4, "a size-4 subset must be present for 4 ids"
    assert len(result4) == 11  # C(4,2)+C(4,3)+C(4,4) = 6+4+1


def test_create_compound_ids_is_subset_of_links_small_terms_to_their_supersets():
    """create_compound_ids's own is_subset_of: names every OTHER compound
    id whose own members is a superset of this entry's own - computed
    only for entries with <=3 members ("Do this only for A with number of
    id1,id2 ... <=3. Empty otherwise" - Ivo), since only a size-<=3 term
    could ever actually substitute into a real (<=3-member) board
    condition. On 3 ids: every pair is a subset of exactly the one triple,
    and the triple itself gets the empty set (also <=3, but nothing
    bigger exists). On 5 ids: a size-4 entry is left at the empty set too,
    by the <=3 rule alone, even though a real superset (the size-5 entry)
    DOES exist and must still be ignored - compound id values are
    arbitrary, so this has to be checked by first mapping members ->
    compound id rather than hardcoding compound id values."""
    ids3 = ("a", "b", "c")
    result3 = create_compound_ids(ids3)
    cid_of = {members: cid for cid, members, _is_subset_of in result3}
    is_subset_of_by_members = {members: is_subset_of for _cid, members, is_subset_of in result3}

    triple_cid = cid_of[("a", "b", "c")]
    assert is_subset_of_by_members[("a", "b")] == {triple_cid}
    assert is_subset_of_by_members[("a", "c")] == {triple_cid}
    assert is_subset_of_by_members[("b", "c")] == {triple_cid}
    assert is_subset_of_by_members[("a", "b", "c")] == set()

    ids5 = ("a", "b", "c", "d", "e")
    result5 = create_compound_ids(ids5)
    for _cid, members, is_subset_of in result5:
        if len(members) == 4:
            assert is_subset_of == set()


def test_generate_compound_candidates_matches_ivos_own_worked_example():
    """generate_compound_candidates (Ivo's own tile-grouped refinement of
    create_compound_ids's generation step): tile a reaches the target
    cell one way (id1), tile b two ways (id2 or id3), tile c two ways
    (id4 or id5). Every individual choice (id1..id5) gets its own
    singleton candidate ("drop the condition 'number of ids in a compound
    >1' ... turn id1, id2 ... the single items into compounds and put
    them on the list" - Ivo), plus one candidate per way to pick >=2
    tiles and one choice from each (Ivo's own hand-listed {a,b1}, {a,b2},
    {b1,c1}, ... {a,b2,c2}) - 17 candidates total. is_subset_of ("also
    fill the "is subset of" - field of the compounds" - Ivo) names every
    OTHER candidate's own members that is a proper superset, same <=3
    cap as create_compound_ids (every candidate here qualifies, the
    biggest is size 3). The full expected result is spelled out literally
    below rather than derived, so every (members, is_subset_of) pair is
    visible directly in the test."""
    tile_choices = {"a": ("id1",), "b": ("id2", "id3"), "c": ("id4", "id5")}
    candidates = generate_compound_candidates(tile_choices)
    assert candidates == {
        (frozenset({"id1"}), frozenset({
            frozenset({"id1", "id2"}), frozenset({"id1", "id3"}),
            frozenset({"id1", "id4"}), frozenset({"id1", "id5"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id2", "id5"}),
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id2"}), frozenset({
            frozenset({"id1", "id2"}), frozenset({"id2", "id4"}), frozenset({"id2", "id5"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id2", "id5"}),
        })),
        (frozenset({"id3"}), frozenset({
            frozenset({"id1", "id3"}), frozenset({"id3", "id4"}), frozenset({"id3", "id5"}),
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id4"}), frozenset({
            frozenset({"id1", "id4"}), frozenset({"id2", "id4"}), frozenset({"id3", "id4"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id3", "id4"}),
        })),
        (frozenset({"id5"}), frozenset({
            frozenset({"id1", "id5"}), frozenset({"id2", "id5"}), frozenset({"id3", "id5"}),
            frozenset({"id1", "id2", "id5"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id1", "id2"}), frozenset({
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id2", "id5"}),
        })),
        (frozenset({"id1", "id3"}), frozenset({
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id1", "id4"}), frozenset({
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id3", "id4"}),
        })),
        (frozenset({"id1", "id5"}), frozenset({
            frozenset({"id1", "id2", "id5"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id2", "id4"}), frozenset({frozenset({"id1", "id2", "id4"})})),
        (frozenset({"id2", "id5"}), frozenset({frozenset({"id1", "id2", "id5"})})),
        (frozenset({"id3", "id4"}), frozenset({frozenset({"id1", "id3", "id4"})})),
        (frozenset({"id3", "id5"}), frozenset({frozenset({"id1", "id3", "id5"})})),
        (frozenset({"id1", "id2", "id4"}), frozenset()),
        (frozenset({"id1", "id2", "id5"}), frozenset()),
        (frozenset({"id1", "id3", "id4"}), frozenset()),
        (frozenset({"id1", "id3", "id5"}), frozenset()),
    }


def test_prune_compound_candidates_discards_candidates_contradicting_a_forcing_relation():
    """prune_compound_candidates: id2 (tile b's b1) forces id4 (tile c's
    c1), so any candidate naming id2 without id4 describes something that
    can never happen - {id2} alone (choosing id2 always drags id4 along),
    {id1,id2} (tile c omitted), {id2,id5} (tile c present with the wrong
    choice), {id1,id2,id5} (same wrong-choice contradiction). The other 13
    survive with their own is_subset_of untouched (still describing their
    relationship to the full, pre-pruning 17-candidate universe - e.g.
    {id1}'s is_subset_of still lists {id1,id2} and {id1,id2,id5} even
    though pruning just removed both of those)."""
    tile_choices = {"a": ("id1",), "b": ("id2", "id3"), "c": ("id4", "id5")}
    candidates = generate_compound_candidates(tile_choices)
    pruned = prune_compound_candidates(candidates, {("id2", "id4")})
    assert pruned == {
        (frozenset({"id1"}), frozenset({
            frozenset({"id1", "id2"}), frozenset({"id1", "id3"}),
            frozenset({"id1", "id4"}), frozenset({"id1", "id5"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id2", "id5"}),
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id3"}), frozenset({
            frozenset({"id1", "id3"}), frozenset({"id3", "id4"}), frozenset({"id3", "id5"}),
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id4"}), frozenset({
            frozenset({"id1", "id4"}), frozenset({"id2", "id4"}), frozenset({"id3", "id4"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id3", "id4"}),
        })),
        (frozenset({"id5"}), frozenset({
            frozenset({"id1", "id5"}), frozenset({"id2", "id5"}), frozenset({"id3", "id5"}),
            frozenset({"id1", "id2", "id5"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id1", "id3"}), frozenset({
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id1", "id4"}), frozenset({
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id3", "id4"}),
        })),
        (frozenset({"id1", "id5"}), frozenset({
            frozenset({"id1", "id2", "id5"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id2", "id4"}), frozenset({frozenset({"id1", "id2", "id4"})})),
        (frozenset({"id3", "id4"}), frozenset({frozenset({"id1", "id3", "id4"})})),
        (frozenset({"id3", "id5"}), frozenset({frozenset({"id1", "id3", "id5"})})),
        (frozenset({"id1", "id2", "id4"}), frozenset()),
        (frozenset({"id1", "id3", "id4"}), frozenset()),
        (frozenset({"id1", "id3", "id5"}), frozenset()),
    }


def test_prune_blocked_compound_candidates_discards_candidates_naming_both_halves_of_a_blocked_pair():
    """prune_blocked_compound_candidates ("check the compound list also
    against the .blocked list. remove blocking items from the compound
    list" - Ivo). Continuing from the forces-pruned 13 candidates above,
    id3 (tile b's b2) and id4 (tile c's c1) are mutually exclusive per
    .blocked - discards {id3,id4} and {id1,id3,id4}, the only two
    surviving candidates naming both. A singleton can never be discarded
    by an ordinary 2-id blocked pair (it only names one id, never both
    halves), so none of the 5 singletons are affected."""
    tile_choices = {"a": ("id1",), "b": ("id2", "id3"), "c": ("id4", "id5")}
    candidates = generate_compound_candidates(tile_choices)
    forced_pruned = prune_compound_candidates(candidates, {("id2", "id4")})
    blocked_pruned = prune_blocked_compound_candidates(forced_pruned, {("id3", "id4")})
    assert blocked_pruned == {
        (frozenset({"id1"}), frozenset({
            frozenset({"id1", "id2"}), frozenset({"id1", "id3"}),
            frozenset({"id1", "id4"}), frozenset({"id1", "id5"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id2", "id5"}),
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id3"}), frozenset({
            frozenset({"id1", "id3"}), frozenset({"id3", "id4"}), frozenset({"id3", "id5"}),
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id4"}), frozenset({
            frozenset({"id1", "id4"}), frozenset({"id2", "id4"}), frozenset({"id3", "id4"}),
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id3", "id4"}),
        })),
        (frozenset({"id5"}), frozenset({
            frozenset({"id1", "id5"}), frozenset({"id2", "id5"}), frozenset({"id3", "id5"}),
            frozenset({"id1", "id2", "id5"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id1", "id3"}), frozenset({
            frozenset({"id1", "id3", "id4"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id1", "id4"}), frozenset({
            frozenset({"id1", "id2", "id4"}), frozenset({"id1", "id3", "id4"}),
        })),
        (frozenset({"id1", "id5"}), frozenset({
            frozenset({"id1", "id2", "id5"}), frozenset({"id1", "id3", "id5"}),
        })),
        (frozenset({"id2", "id4"}), frozenset({frozenset({"id1", "id2", "id4"})})),
        (frozenset({"id3", "id5"}), frozenset({frozenset({"id1", "id3", "id5"})})),
        (frozenset({"id1", "id2", "id4"}), frozenset()),
        (frozenset({"id1", "id3", "id5"}), frozenset()),
    }


def _tagged(*ids):
    return frozenset((id_, PlacementStatus.unknown) for id_ in ids)


def test_compoundize_forced_by_replaces_every_term_with_singleton_compounds():
    """compoundize_forced_by (added 2026-11-01, Ivo: "now replace all ids,
    tuple terms and triple terms in .forced_by and .forces in the 3x3 of
    SquareItems by compounds if possible"). No bigger compound exists yet
    for {p,q,r} (the triple term below), so "if possible" falls back to
    wrapping each of p, q, r in its own singleton compound, preserving
    the term's own arity (3 compounds, not 1) - while id8's two separate
    size-1 terms each collapse to a single compound too, REUSING q's and
    r's own singleton compounds (the same ones minted for id6's own
    triple), since get_or_create_compound interns by members. .forces is
    cleared, not rebuilt - see the function's own docstring for why."""
    m = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            m[i, j] = SquareItem()
    p, q, r = (0, 0), (0, 1), (0, 2)
    id6, id7, id8 = (1, 0), (1, 1), (1, 2)
    m[id6].forced_by = {_tagged(p, q, r)}
    m[id7].forced_by = {_tagged(p)}
    m[id8].forced_by = {_tagged(q), _tagged(r)}

    realm = ((0, 2), (0, 3))
    compounds = compoundize_forced_by(m, realm)

    assert len(compounds) == 3  # one singleton per real id - p, q, r
    cid_p = compounds[frozenset({p})]
    cid_q = compounds[frozenset({q})]
    cid_r = compounds[frozenset({r})]
    assert len({cid_p, cid_q, cid_r}) == 3  # every singleton compound is distinct

    assert m[id6].forced_by == {_tagged(cid_p, cid_q, cid_r)}
    assert m[id7].forced_by == {_tagged(cid_p)}
    assert m[id8].forced_by == {_tagged(cid_q), _tagged(cid_r)}
    assert m[id6].forces == set()
    assert m[id7].forces == set()
    assert m[id8].forces == set()


def test_propagate_compound_links_creates_union_compound_and_inherits_forces():
    """propagate_compound_links (added 2026-11-01), continuing directly
    from compoundize_forced_by's own result above - Ivo's own worked
    illustration: id6's forced_by term (A, B, C) (A/B/C the singleton
    compounds for p/q/r) gets a new compound Q = union(A, B, C) = {p,q,r},
    with "the such-created new link" credited to Q alone: Q gains id6 in
    its own compound_forces entry directly. Separately (pass 1, from the
    already-compoundized .forced_by alone, before any union): A
    participates in id7's own direct term ("A forces id7"), and B/C each
    participate in one of id8's own two terms ("B forces id8, C forces
    id8"). Q, being a strict superset of A, B, AND C, inherits every one
    of their own compound_forces entries too - "Q forces what its subsets
    (A,B,C, and others) force, but in addition the path through id6" -
    giving Q exactly {id6, id7, id8}, not just {id6}. The new Q link is
    also recorded back into id6's own .forced_by, alongside (A,B,C) -
    one is not replaced by the other."""
    m = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            m[i, j] = SquareItem()
    p, q, r = (0, 0), (0, 1), (0, 2)
    id6, id7, id8 = (1, 0), (1, 1), (1, 2)
    m[id6].forced_by = {_tagged(p, q, r)}
    m[id7].forced_by = {_tagged(p)}
    m[id8].forced_by = {_tagged(q), _tagged(r)}

    realm = ((0, 2), (0, 3))
    compounds = compoundize_forced_by(m, realm)
    cid_p = compounds[frozenset({p})]
    cid_q = compounds[frozenset({q})]
    cid_r = compounds[frozenset({r})]

    compound_forces = propagate_compound_links(m, realm, compounds)

    assert compound_forces[cid_p] == {id6, id7}
    assert compound_forces[cid_q] == {id6, id8}
    assert compound_forces[cid_r] == {id6, id8}

    assert frozenset({p, q, r}) in compounds, "Q must have been minted"
    cid_q_union = compounds[frozenset({p, q, r})]
    assert cid_q_union not in {cid_p, cid_q, cid_r}
    assert compound_forces[cid_q_union] == {id6, id7, id8}

    assert m[id6].forced_by == {
        _tagged(cid_p, cid_q, cid_r),
        _tagged(cid_q_union),
    }
