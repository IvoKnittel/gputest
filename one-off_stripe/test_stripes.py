from pickletools import uint8
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def test_load():
    image_path = 'image_monochrom.bmp'
    image = Image.open(image_path)
    image = image.convert('1')
    np_array = np.array(image).astype(np.uint8)
    plt.imshow(np_array, cmap='gray')
    plt.axis('on')
    plt.show()