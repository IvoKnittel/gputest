import numpy as np
import matplotlib.pyplot as plt

v = 0
h = 1


def display_superposition(a, b):
    # Example arrays (replace with your own)
    A = a.astype(np.uint8)
    B = b.astype(np.uint8)

    # Normalize to [0,1] for alpha
    A_alpha = A.astype(float) / 255.0
    B_alpha = B.astype(float) / 255.0

    # Create RGBA images
    A_rgba = np.zeros((A.shape[v], A.shape[h], 4), dtype=float)
    B_rgba = np.zeros((B.shape[v], B.shape[h], 4), dtype=float)

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


def image_generator(number, method, seed=None):
    """Yield `number` synthetic test images as (binary_image, image_noisy_array) pairs.

    method selects how each image is built - only 'noisy_square' exists for now: a
    40x40 field of 1s with a 15x15 zero square in the middle, scaled to the uint8
    range and perturbed with per-pixel noise. seed feeds a numpy Generator used for
    that noise, so runs are reproducible; pass None for non-reproducible noise. Not
    every future method will need randomness - seed only applies where it does.
    """
    if method == 'noisy_square':
        rng = np.random.default_rng(seed)
        for _ in range(number):
            binary_image = np.ones((40, 40))
            binary_image[7:22, 7:22] = 0
            binary_array = np.array(binary_image).astype(np.uint8)
            binary_array_u8_range = binary_array * 200 + 20
            noise = rng.choice([-1, 0, 1], size=binary_array_u8_range.shape)
            image_noisy_array = np.clip(binary_array_u8_range + noise, 0, 255).astype(np.uint8)
            yield binary_image, image_noisy_array
        return

    raise ValueError(f"unknown image_generator method: {method!r}")
