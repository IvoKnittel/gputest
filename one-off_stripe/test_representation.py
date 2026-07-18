import matplotlib.pyplot as plt

from representation import (map_of_squares_from_array,
                             display_map_of_squares_3States,
                             real_space_map)


def test_map_input_display():
    grid = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]]

    m = map_of_squares_from_array(grid)

    fig, (ax_map, ax_real) = plt.subplots(1, 2, figsize=(10, 5))

    ax_map.imshow(display_map_of_squares_3States(m), cmap='gray')
    ax_map.set_title('map_of_squares')
    ax_map.axis('on')

    ax_real.imshow(real_space_map(m), cmap='gray')
    ax_real.set_title('real space')
    ax_real.axis('on')

    plt.tight_layout()
    plt.show()
