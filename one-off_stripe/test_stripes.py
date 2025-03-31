from image_to_squares import image_squares
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def test_load():
    image_path = 'image_monochrom.bmp'
    image = Image.open(image_path)
    image = image.convert('1')
    image_array = np.array(image).astype(np.uint8)
    image2x2 = image_squares(image_array)

    plt.imshow(image_array, cmap='gray')
    plt.axis('on')
    plt.show()