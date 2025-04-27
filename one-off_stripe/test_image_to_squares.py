import numpy as np

from image_to_squares import (image_squares,
                              image_squares_ranked0,
                              image_squares_ranked,
                              image_squares_select)

from PIL import Image
import matplotlib.pyplot as plt


def test_image_to_squares():
    image_path = 'image_monochrom.bmp'
    image = Image.open(image_path)
    image = image.convert('1')
    image_array = np.array(image).astype(np.uint8)
    image2x2 = image_squares(image_array)

    plt.imshow(image_array, cmap='gray')
    plt.axis('on')
    plt.show()

    r= image_squares_ranked0(image2x2)
    square_storage_location_map=image_squares_ranked(r)
    square_extension_map = -np.ones((2*square_storage_location_map.shape[0], 2*square_storage_location_map.shape[1]), dtype=int)
    square_extension_map, square_storage_location_map= image_squares_select(square_extension_map, square_storage_location_map)