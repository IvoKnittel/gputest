import numpy as np
import numpy.typing as npt
from item import Item
import math


def load_image(image):
    for i in range(0,image.shape[0]):
        for j in range(0, image.shape[1]):
            pixel = image[i,j]

