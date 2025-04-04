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

def image_squares_ranked0(image_squares):
    squares_ranked0 = np.zeros((2*image_squares.shape[0],2*image_squares.shape[1]), dtype=int)
    for i in range(1, image_squares.shape[0]):
        for j in range(1, image_squares.shape[1]):

            qvec=np.zeros(4, dtype=float)
            qvec[0] = image_squares[i - 1, j - 1].quality
            qvec[1] = image_squares[i - 1, j].quality
            qvec[2] = image_squares[i, j - 1].quality
            qvec[3] = image_squares[i, j].quality
            indices = np.argsort(qvec)
            q=(2*i,2*j)
            squares_ranked0[q[0], q[1]] = indices[0]
            squares_ranked0[q[0], q[1] + 1] = indices[1]
            squares_ranked0[q[0] + 1, q[1]] = indices[2]
            squares_ranked0[q[0] + 1, q[1] + 1] = indices[3]

    return squares_ranked0

def image_squares_ranked(r):
    s = np.zeros(r.shape, dtype=float)
    for i in range(2, r.shape[0]-2):
        for j in range(2, r.shape[1]-2):
            s[2 * i, 2 * j] = np.mean([r[2 * (i - 1) + 1, 2 * (j - 1) + 1], r[2 * (i - 1) + 1, 2 * j], r[2 * i, 2 * (j - 1)], r[2 * i, 2 * j + 1]])
            s[2 * i, 2 * j + 1] = np.mean([r[2 * (i - 1) + 1, 2 * j + 1], r[2 * (i - 1) + 1, 2 * (j + 1)], r[2 * i, 2 * j],r[2 * i, 2 * (j + 1) + 1]])
            s[2 * i + 1, 2 * j] = np.mean([r[2 * i + 1, 2 * (j - 1) + 1], r[2 * i + 1, 2 * j], r[2 * (i + 1), 2 * (j - 1)],r[2 * (i + 1), 2 * j + 1]])
            s[2 * i + 1, 2 * j + 1] = np.mean([r[2 * i + 1, 2 * j], r[2 * i + 1, 2 * (j + 1)], r[2 * (i + 1), 2 * j], [2 * (i + 1), 2 * (j + 1)]])
    return s
#  def image_rank_quA(image_squares):