import numpy as np
import  pytest
from item import Item, random_items
from rowmergeinputs import I0
from rowmergebasics import A0

def pytest_addoption(parser):
    parser.addoption(
        "--do_display", action="store_true", default=False, help="Enable display of plots"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "do_display: mark test to enable display of plots")


@pytest.fixture(scope="module")
def items():
    return random_items(20)

@pytest.fixture(scope="module")
def sz(items):
    return len(items)

@pytest.fixture(scope="module")
def merged_items(items, sz):
    if sz < 2:
        raise ValueError("The input list must contain at least two items.")

    m = np.empty((2, sz), dtype=Item)

    m[0, 0] = Item()
    m[1, 0] = Item(items[0], items[1])

    for idx in range(1, sz - 1):
        m[0, idx] = Item(items[idx - 1], items[idx])
        m[1, idx] = Item(items[idx], items[idx + 1])

    m[0, -1] = Item(items[-2], items[-1])
    m[1, -1] = Item()

    return m

@pytest.fixture(scope="module")
def inputs(items):
    return I0(items)


@pytest.fixture(scope="module")
def basics(inputs, sz):
    b = np.full((2, sz), 0)
    b[0,:]=inputs.prefer_vector
    b[1,:]=inputs.vote_vector
    return A0(inputs)

