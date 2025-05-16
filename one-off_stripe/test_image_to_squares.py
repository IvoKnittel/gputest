import numpy as np

from image_to_squares import (image_squares,
                              image_squares_quality,
                              image_squares_ranked0,
                              image_squares_ranked,
                              image_squares_select)

from PIL import Image
import matplotlib.pyplot as plt

def display_superposition(a, b):

    # Example arrays (replace with your own)
    A = a.astype(np.uint8)
    B = b.astype(np.uint8)

    # Normalize to [0,1] for alpha
    A_alpha = A.astype(float) / 255.0
    B_alpha = B.astype(float) / 255.0

    # Create RGBA images
    A_rgba = np.zeros((A.shape[0], A.shape[1], 4), dtype=float)
    B_rgba = np.zeros((B.shape[0], B.shape[1], 4), dtype=float)

    # Black for A
    A_rgba[..., 0:3] = 0  # RGB = 0,0,0
    A_rgba[..., 3] = A_alpha  # Alpha

    # Green for B
    B_rgba[..., 0] = 0  # R
    B_rgba[..., 1] = 1  # G
    B_rgba[..., 2] = 0  # B
    B_rgba[..., 3] = B_alpha  # Alpha

    plt.figure(figsize=(6, 6))
    plt.imshow(A_rgba)
    plt.imshow(B_rgba)
    plt.axis('off')
    plt.show()


def test_image_to_squares():
    image_path = 'image2_monochrom.bmp'
    image = Image.open(image_path)
    binary_image = np.ones((10,10))
    binary_image[2:6,2:6]=0
    binary_array = np.array(binary_image).astype(np.uint8)
    binary_array_u8_range = binary_array * 200 + 20  # Low value = 1, High value = 254

    # Generate random noise (-1, 0, 1)
    noise = np.random.choice([-1, 0, 1], size=binary_array_u8_range.shape)

    # Add noise and clip to uint8 range
    image_noisy_array = np.clip(binary_array_u8_range + noise, 0, 255).astype(np.uint8)
    image2x2 = image_squares(image_noisy_array)

    #plt.imshow(image_noisy_array, cmap='gray')
    #plt.axis('on')
    #plt.show()
    squares_quality = image_squares_quality(image2x2)

    #plt.imshow(squares_quality, cmap='gray')
    #plt.axis('on')
    #plt.show()

    r= image_squares_ranked0(squares_quality)
    #plt.imshow(r, cmap='gray')
    #plt.axis('on')
    #plt.show()

    square_storage_location_map=image_squares_ranked(r)
    #plt.imshow(square_storage_location_map, cmap='gray')
    #plt.axis('on')
    #plt.show()

    square_extension_map, square_storage_location_map= image_squares_select(square_storage_location_map)
    #plt.imshow(square_extension_map, cmap='gray')
    #plt.axis('on')
    #plt.show()
    plt.imshow(square_extension_map, cmap='gray')
    plt.axis('on')
    plt.show()
    plt.imshow(square_storage_location_map, cmap='gray')
    plt.axis('on')
    plt.show()