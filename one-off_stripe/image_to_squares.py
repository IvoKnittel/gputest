import numpy as np
from item import Item

def expand4(n):
    num_tiles = int(np.ceil/4)
    return 4*num_tiles , num_tiles

def image_squares(image):
    height_expanded, num_tiles_vertical = expand4(image.shape[0])
    width_expanded, num_tiles_horizontal = expand4(image.shape[1])

    image_expanded = np.empty((height_expanded, width_expanded), dtype=Item)
    image_expanded[0:image.shape[0],0:image.shape[1]]=image
    image_1x2_items = np.empty((height_expanded, width_expanded), dtype=Item)
    for i in range(0, num_tiles_vertical-1):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_1x2_items[m,n+k] = Item(Item(image_expanded[m,n+k]), Item(image_expanded[m+1,n+k]))
                image_1x2_items[m+1,n+k] = Item(Item(image_expanded[m+1,n+k]), Item(image_expanded[m+2,n+k]))

    for i in range(0, num_tiles_vertical-1):
        for j in range(0, num_tiles_horizontal):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_1x2_items[m+2,n+k] = Item(Item(image_expanded[m+2,n+k]), Item(image_expanded[m+3,n+k]))
                image_1x2_items[m+3,n+k] = Item(Item(image_expanded[m+3,n+k]), Item(image_expanded[m+4,n+k]))


    image_2x2_items = np.empty((height_expanded+1, width_expanded+1), dtype=Item)
    for i in range(1, num_tiles_vertical):
        for j in range(1, num_tiles_horizontal-1):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x2_items[1+m+k,1+n] = Item(image_1x2_items[m+k,n]), Item(image_1x2_items[m+k,n+1])
                image_2x2_items[1+m+k,1+n+1] = Item(image_1x2_items[m+k,n+1]), Item(image_1x2_items[m+k,n+2])

    for i in range(0, num_tiles_vertical):
        for j in range(0, num_tiles_horizontal-1):
            for k in range(0,4):
                m=4*i
                n=4*j
                image_2x2_items[1+m+k,1+n+2] = Item(image_1x2_items[m+k,n+2]), Item(image_1x2_items[m+k,n+3])
                image_2x2_items[1+m+k,1+n+3] = Item(image_1x2_items[m+k,n+3]), Item(image_1x2_items[m+k,n+4])

    return image_2x2_items

def image_squares_ranked0(image_squares):
    squares_ranked0 = np.zeros((2*(image_squares.shape[0]-1),2*(image_squares.shape[1])), dtype=int)
    for i in range(1, image_squares.shape[0]):
        for j in range(1, image_squares.shape[1]):

            qvec=np.zeros(4, dtype=float)
            qvec[0] = image_squares[i-1, j-1].quality
            qvec[1] = image_squares[i-1, j].quality
            qvec[2] = image_squares[i, j-1].quality
            qvec[3] = image_squares[i, j].quality
            ranks = np.argsort(qvec)
            q=(2*i,2*j)
            squares_ranked0[q[0], q[1]] = ranks[0]
            squares_ranked0[q[0], q[1] + 1] = ranks[1]
            squares_ranked0[q[0] + 1, q[1]] = ranks[2]
            squares_ranked0[q[0] + 1, q[1] + 1] = ranks[3]

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