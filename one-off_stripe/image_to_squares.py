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
    squares_ranked0 = np.zeros((2*(image_squares.shape[0]-1),2*(image_squares.shape[1]-1)), dtype=int)
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

def pad_ranked_squares(m):
    r0 = m % 6
    if r0 < 4:
        return 4 - r0

    if r0 == 5:
        return 1

    return 0

def image_squares_ranked(r):
    e = (pad_ranked_squares(r.shape[0]),pad_ranked_squares(r.shape[0]))
    s = -np.ones((r.shape[0]+e[0],r.shape[1]+e[1]), dtype=int)
    for i in range(2, r.shape[0]-2):
        for j in range(2, r.shape[1]-2):
            M = np.iinfo(np.int32).max/3 #3 is the max rank value
            s[2 * i, 2 * j] = int(M*np.mean([r[2 * (i - 1) + 1, 2 * (j - 1) + 1], r[2 * (i - 1) + 1, 2 * j], r[2 * i, 2 * (j - 1)], r[2 * i, 2 * j + 1]]))
            s[2 * i, 2 * j + 1] = int(M*np.mean([r[2 * (i - 1) + 1, 2 * j + 1], r[2 * (i - 1) + 1, 2 * (j + 1)], r[2 * i, 2 * j],r[2 * i, 2 * (j + 1) + 1]]))
            s[2 * i + 1, 2 * j] = int(M*np.mean([r[2 * i + 1, 2 * (j - 1) + 1], r[2 * i + 1, 2 * j], r[2 * (i + 1), 2 * (j - 1)],r[2 * (i + 1), 2 * j + 1]]))
            s[2 * i + 1, 2 * j + 1] = int(M*np.mean([r[2 * i + 1, 2 * j], r[2 * i + 1, 2 * (j + 1)], r[2 * (i + 1), 2 * j], [2 * (i + 1), 2 * (j + 1)]]))
    return s

def size_(n):
    r= n%6
    N = np.ceil((n-1)/6)
    M = np.ceil((n + 3) / 6)
    return (max(3+6*N, 6*M),N,M)

def image_squares_select(s):
    h, H0, H1 = size_(s.shape[0])
    w, W0, W1 = size_(s.shape[1])
    sz= (h,w)
    q = -np.ones(sz, dtype=int)
    q[3:3+s.shape[0],3:3+s.shape[1]]=s
    for I in range(0,H0):
        for J in range(0, W0):
            r=s[3+6*I:3+6*(I+1),3+6*J:3+6*(J+1)]
            mx = -np.ones((0,4),dtype=int)
            mi =  np.empty((0,4),dtype=(int,int))
            for k in range(0,6):
                for l in range(0, 6):
                    v=r[k,l]
                    if v>mx[0]:
                        mx[0]=v
                        mi[0]=(k,l)
                    elif v>mx[1]:
                        mx[1]=v
                        mi[1]=(k,l)
                    elif v>mx[2]:
                        mx[2]=v
                        mi[2]=(k,l)
                    elif v>mx[3]:
                        mx[3]=v
                        mi[3]=(k,l)

    for I in range(0,H1):
        for J in range(0, W1):
            r=s[6*I:6*(I+1),6*J:6*(J+1)]
            mx = -np.ones((0,4),dtype=int)
            mi =  np.empty((0,4),dtype=(int,int))
            for k in range(0,6):
                for l in range(0, 6):
                    v=r[k,l]
                    if v>mx[0]:
                        mx[0]=v
                        mi[0]=(k,l)
                    elif v>mx[1]:
                        mx[1]=v
                        mi[1]=(k,l)
                    elif v>mx[2]:
                        mx[2]=v
                        mi[2]=(k,l)
                    elif v>mx[3]:
                        mx[3]=v
                        mi[3]=(k,l)

