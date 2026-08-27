import pytest
import matplotlib.pyplot as plt

from closure import do_closure
from image_to_squares import build_quality_map, image_squares_select_single, tile_counts_2d


@pytest.fixture
def quality_map_setup():
    return build_quality_map()


def test_image_to_squares(quality_map_setup):
    binary_image, image_noisy_array, square_map = quality_map_setup

    plt.imshow(image_noisy_array, cmap='gray')
    plt.axis('on')
    plt.show()


def test_square_placement(quality_map_setup):
    binary_image, image_noisy_array, square_map = quality_map_setup

    do_closure(square_map, "test_square_placement")

    num_tiles_expand_noshift_shift = tile_counts_2d(square_map.shape)
    for shift in ([0, 0], [0, 1], [1, 0], [1, 1]):
        square_map = image_squares_select_single(square_map, num_tiles_expand_noshift_shift, shift)
