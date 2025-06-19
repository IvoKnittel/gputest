import numpy as np

from image_to_squares import (image_squares,
                              image_squares_quality,
                              image_squares_ranked0,
                              image_squares_ranked,
                              size_expand2d,
                              image_squares_select_single,
                              tile_display_single)

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
    binary_image = np.ones((40,40))
    binary_image[7:22,7:22]=0
    binary_array = np.array(binary_image).astype(np.uint8)
    binary_array_u8_range = binary_array * 200 + 20  # Low value = 1, High value = 254

    # Generate random noise (-1, 0, 1)
    noise = np.random.choice([-1, 0, 1], size=binary_array_u8_range.shape)

    # Add noise and clip to uint8 range
    image_noisy_array = np.clip(binary_array_u8_range + noise, 0, 255).astype(np.uint8)
    image2x2, image1x2, image2x1 = image_squares(image_noisy_array)
    images=(image2x2, image1x2, image2x1)
    plt.imshow(image_noisy_array, cmap='gray')
    plt.axis('on')
    plt.show()
    squares_quality = image_squares_quality(image2x2)

    # plt.imshow(squares_quality, cmap='gray')
    # plt.axis('on')
    # plt.show()

    r= image_squares_ranked0(squares_quality)
    #plt.imshow(r, cmap='gray')
    #plt.axis('on')
    #plt.show()

    square_storage_location_map = image_squares_ranked(r)
    szl=square_storage_location_map.shape
    sze=(szl[0]+1,szl[1]+1)
    # plt.imshow(square_storage_location_map, cmap='gray')
    # plt.axis('on')
    # plt.show()
    sz_halftile=3
    sz_expand, num_tiles_expand = size_expand2d(square_storage_location_map.shape)
    square_storage_location_map_expand = -np.ones(sz_expand, dtype=float)
    square_storage_location_map_expand[sz_halftile:sz_halftile + square_storage_location_map.shape[0],sz_halftile:sz_halftile + square_storage_location_map.shape[1]] = square_storage_location_map
    square_extension_map_expand = -np.zeros((square_storage_location_map_expand.shape[0]+1, square_storage_location_map_expand.shape[1]+1), dtype=int)

    # square_extension_map_expand = tile_display(square_extension_map_expand, num_tiles_expand, (0,0),(1,1), (1,2))
    #square_storage_location_map_expand, square_extension_map_expand = image_squares_select(square_storage_location_map_expand, square_extension_map_expand, num_tiles_expand, (0,0),(1,1))
    square_storage_location_map_expand, square_extension_map_expand = image_squares_select_single(square_storage_location_map_expand, square_extension_map_expand, num_tiles_expand, (0,0),False, images)
    square_storage_location_map_expand, square_extension_map_expand = image_squares_select_single(square_storage_location_map_expand, square_extension_map_expand, num_tiles_expand,(1,1),False, images)

    #square_extension_map = square_extension_map_expand[sz_halftile:sz_halftile + sze[0],sz_halftile:sz_halftile + sze[1]]
    #square_storage_location_map = square_storage_location_map_expand[sz_halftile:sz_halftile + szl[0],sz_halftile:sz_halftile + szl[1]]
    # plt.imshow(square_storage_location_map, cmap='gray')
    # plt.axis('on')
    # plt.show()

    # square_extension_map_expand = tile_display_single(square_extension_map_expand, num_tiles_expand, (0,0), 3)
    # square_extension_map_expand = tile_display_single(square_extension_map_expand, num_tiles_expand,(0,0),4)

    #square_extension_map = square_extension_map_expand[sz_halftile:sz_halftile + sze[0],sz_halftile:sz_halftile + sze[1]]
    # plt.imshow(square_extension_map, cmap='gray')
    # plt.axis('on')
    # plt.show()
    square_storage_location_map_expand, square_extension_map_expand = image_squares_select_single(square_storage_location_map_expand, square_extension_map_expand, num_tiles_expand, (1,0), True, images)
    square_storage_location_map_expand, square_extension_map_expand = image_squares_select_single(square_storage_location_map_expand, square_extension_map_expand, num_tiles_expand,(0,1), True, images)


    square_extension_map = square_extension_map_expand[sz_halftile:sz_halftile + sze[0],sz_halftile:sz_halftile + sze[1]]
    # square_storage_location_map = square_storage_location_map_expand[sz_halftile:sz_halftile + szl[0],sz_halftile:sz_halftile + szl[1]]
    # plt.imshow(square_storage_location_map, cmap='gray')
    # plt.axis('on')
    # plt.show()

    plt.imshow(square_extension_map[0:40,0:40]+binary_array, cmap='gray')
    plt.axis('on')
    plt.show()