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
def inputs(items):
    return I0(items)


@pytest.fixture(scope="module")
def basics(inputs, sz):
    b = np.full((2, sz), 0)
    b[0,:]=inputs.prefer_vector
    b[1,:]=inputs.vote_vector
    return A0(inputs)

