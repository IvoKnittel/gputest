import matplotlib.pyplot as plt
import math
from item import random_items, odd_step, calendar_items, sine_items, chirp_items
from rowmerge import items_for_display, tree
# @pytest.fixture(autouse=True)
# def set_do_display(pytestconfig):
#     global do_display
#     do_display = pytestconfig.getoption("do_display")


# def pytest_addoption(parser):
#     parser.addoption(
#         "--do_display", action="store_true", default=False, help="Enable display of plots"
#     )
#
# def pytest_configure(config):
#     config.addinivalue_line("markers", "do_display: mark test to enable display of plots")


def tree_visual(base_items, tree_data, num_gens, start_plot_gen):
    x, a = items_for_display(base_items, len(base_items))
    plt.figure(1)
    plt.plot(x, a, 'o', color='red')
    colors = ['blue', 'green', 'black', 'cyan', 'gray', 'magenta']
    for g in range(num_gens):
        if g >= start_plot_gen:
            crt_data = tree_data[g]
            plt.plot(crt_data[0], crt_data[1]+(g-start_plot_gen)*0.02, 'o', color=colors[g-start_plot_gen])
        # plt.plot(x2, a2 + s2/10, label='a2 + s2', linestyle='--')
        # plt.plot(x2, a2 - s2/10, label='a2 - s2', linestyle='--')

    plt.xlabel('x')
    plt.ylabel('Values')
    plt.title('Plot of a, a2, and a2 ± s2 versus x')
    plt.show()


def test_tree_visual_sine():
    base_items = sine_items()
    num_gens = 5
    start_plot_gen = 3
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)

def test_tree_visual_calendar():
    base_items = calendar_items()
    num_gens = 5
    start_plot_gen = 3
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)


def test_tree_visual_step():
    base_items = odd_step(20)
    num_gens = 3
    start_plot_gen = 0
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)

def test_tree_visual_random():
    base_items = random_items(20)
    num_gens = 3
    start_plot_gen = 0
    assert num_gens < math.floor(math.log(len(base_items)))
    assert num_gens > start_plot_gen
    tree_data = tree(base_items, num_gens)
    tree_visual(base_items, tree_data, num_gens, start_plot_gen)
