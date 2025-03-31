import numpy as np
from item import Item

def crop(n):
    num_tiles = int(np.floor(n-2)/4)
    return 4*num_tiles + 2 , num_tiles

def image_squares(image):
    height_cropped, num_tiles_vertical = crop(image.shape[0])
    width_cropped, num_tiles_horizontal = crop(image.shape[1])

    image_cropped = image((height_cropped,width_cropped))
    image_1x2_items = np.empty((height_cropped, width_cropped), dtype=Item)
    for i in range(0, 2*num_tiles_vertical + 1):
        for j in range(0, width_cropped):
            image_1x2_items[2*i,j] = Item(Item(image_cropped[2*i,j]), Item(image_cropped[2*i+1,j]))

    for i in range(0, 2*num_tiles_vertical):
        for j in range(0, width_cropped):
            image_1x2_items[2*i+1,j] = Item(Item(image_cropped[2*i+1,j]), Item(image_cropped[2*i+2,j]))

    image_2x2_items = np.empty((height_cropped, width_cropped), dtype=Item)
    for i in range(0, height_cropped):
        for j in range(0, 2*num_tiles_horizontal+1):
            image_2x2_items[i,2*j] = Item(image_1x2_items[2*i,j]), Item(image_1x2_items[2*i+1,j])

    for i in range(0, height_cropped):
        for j in range(0, 2*num_tiles_horizontal):
            image_2x2_items[i,2*j+1] = Item(image_1x2_items[2*i+1,j]), Item(image_1x2_items[2*i+2,j])