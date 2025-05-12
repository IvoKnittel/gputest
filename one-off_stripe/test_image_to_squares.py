import numpy as np

from image_to_squares import (image_squares,
                              image_squares_ranked0,
                              image_squares_ranked,
                              image_squares_select)

from PIL import Image
import matplotlib.pyplot as plt


def test_image_to_squares():
    image_path = 'image2_monochrom.bmp'
    image = Image.open(image_path)
    binary_image = image.convert('1')
    binary_array = np.array(binary_image).astype(np.uint8)
    binary_array_u8_range = binary_array * 253 + 1  # Low value = 1, High value = 254


    # Generate random noise (-1, 0, 1)
    noise = np.random.choice([-1, 0, 1], size=binary_array_u8_range.shape)

    # Add noise and clip to uint8 range
    image_noisy_array = np.clip(binary_array_u8_range + noise, 0, 255).astype(np.uint8)
    image2x2 = image_squares(binary_array_u8_range)

    plt.imshow(image_noisy_array, cmap='gray')
    plt.axis('on')
    plt.show()

    r= image_squares_ranked0(image2x2)
    plt.imshow(r, cmap='gray')
    plt.axis('on')
    plt.show()

    square_storage_location_map=image_squares_ranked(r)
    plt.imshow(square_storage_location_map, cmap='gray')
    plt.axis('on')
    plt.show()
    # square_extension_map = -np.ones((2*square_storage_location_map.shape[0], 2*square_storage_location_map.shape[1]), dtype=int)
    #square_extension_map, square_storage_location_map= image_squares_select(square_extension_map, square_storage_location_map)
    #plt.imshow(square_extension_map, cmap='gray')
    #plt.axis('on')
    #plt.show()