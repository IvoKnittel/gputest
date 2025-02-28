import pytest
import matplotlib.pyplot as plt
import numpy as np
from rowmerge import *


@pytest.fixture(autouse=True)
def set_do_display(pytestconfig):
    global do_display
    do_display = pytestconfig.getoption("do_display")


def display_plot(title, x, y):
    if do_display:
        plt.figure()
        plt.plot(x, y, 'o-')
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()


def get_merges_begin_inital(merges_initial):
    sz = len(merges_initial)
    merges_begin_inital = np.array(["x"] * sz, dtype=str)

    if merges_initial[0] == 'c':
        merges_begin_inital[0] = 'c'
    for j in range(1, sz - 1):
        if merges_initial[j] == 'c' and merges_initial[j - 1] != 'c':
            merges_begin_inital[j] = 'c'

    if merges_initial[0] == 't':
        merges_begin_inital[0] = 't'
    for j in range(1, sz - 1):
        if merges_initial[j] == 't' and merges_initial[j - 1] != 't':
            merges_begin_inital[j] = 't'


def check_letter_repeats(sequence: str, sequence_begin: str, letter: str, block_length: int):
    # merge_type: either "s", "c", or "t"
    sz = len(sequence)
    assert block_length < sz
    assert len(letter) == 1
    assert len(sequence) == len(sequence_begin)
    for j in range(0, sz):
        if sequence_begin[j] == letter:
            for k in range(block_length):
                assert sequence[j + k] == letter


def check_singles_occur_isolated(sequence: str):
    # merge_type: either "s", "c", or "t"
    sz = len(sequence)
    if sequence[0] == 's':
        assert sequence[0] != 's'
    if sequence[-1] == 's':
        assert sequence[-2] != 's'

    for j in range(1, sz - 1):
        if sequence[j] == 's':
            assert sequence[j - 1] != 's' and sequence[j + 1] != 's'

def test_merges_initial_are_disjunct(merges_initial):
    check_singles_occur_isolated(merges_initial)
''.join(begin_couples_initial)


def test_merges_initial(merges_initial: str):
    check_singles_occur_isolated(merges_initial)
    merges_begin_inital = get_merges_begin_inital(merges_initial)

    check_letter_repeats(merges_initial, merges_begin_inital, 'c', 2)
    check_letter_repeats(merges_initial, merges_begin_inital, 't', 3)

def test_step_D_all_items_classified():
    classification = ['Single', 'Triple', 'Single', 'Single']
    assert all_items_classified(classification)

def test_step_E_chain_rules():
    classification = ['Single', 'Triple', 'Single']
    assert check_chain_rules(classification)

def test_step_F_terminated_chain():
    classification = ['Single', 'Triple', 'Single']
    assert check_terminated_chain(classification)

def test_step_G_convert_terminated_chain():
    classification = ['Single', 'Triple', 'Single']
    expected_conversion = ['Single', 'Couple', 'Single']

    conversion = convert_terminated_chain(classification)
    assert conversion == expected_conversion

    display_plot("Step G: Convert Terminated Chain", list(range(len(conversion))), conversion)

def test_step_H_all_items_classified_after_conversion():
    classification = ['Single', 'Couple', 'Single']
    assert all_items_classified(classification)

def test_step_I_displacement():
    classification = ['Single', 'Couple', 'Single']
    expected_displacement = [0, 1, -1]

    displacement = calculate_displacement(classification)
    assert displacement == expected_displacement

    display_plot("Step I: Displacement", list(range(len(displacement))), displacement)

def test_step_J_length_matches():
    classification = ['Single', 'Couple', 'Single']
    n = 2
    assert length_matches(classification, n)

def test_step_K_enforce_displacement():
    displacement = [0, 1, -1]
    assert enforce_displacement(displacement)