#    Finish 2D convolution/filtering by your self. 
#    What you are supposed to do can be described as "median blur", which means by using a sliding window 
#    on an image, your task is not going to do a normal convolution, but to find the median value within 
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
#    with O(W*H*m*n log(m*n)) to O(W*H*m*n*m*n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#
#    Python version:
#    def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#        Please finish your code under this blank

"""
20190717 by Qing Peng

"""

import numpy as np
import cv2
import statistics

def medianBlur(img, kernel, padding_way):
    """
    img: list of list
    kernel: list of list
    padding_way: string
    """

    # Get image size and kernel size
    W, H = img.shape[:2]
    m, n = kernel.shape[:2]

    #if m%2 == 0 or n%2 == 0:
    #    return None

    # Size of augmented image with padding = W + 2*(m-1)/2, H + 2*(n-1)/2
    
    # Padding
    a = (m - 1) / 2
    b = (n - 1) / 2

    W_aug = W + 2*a
    H_aug = H + 2*b

    # Initialization of the augmented padding image with all 0
    img_aug = [[0 for x in range(W_aug)] for y in range(H_aug)]
    # Original image in the center
    img_aug[b:H_aug-b-1][a:W_aug-a-1] = img[:][:]

    if padding_way == "ZERO":
        pass
    elif padding_way == "REPLICA":
        # 4 corners
        img_aug[0:b-1][0:a-1] = img[0][0]
        img_aug[0:b-1][W_aug-a:W_aug-1] = img[0][W-1]
        img_aug[H_aug-b:H_aug-1][0:a-1] = img[H-1][0]
        img_aug[H_aug-b:H_aug-1][W_aug-a:W_aug-1] = img[H-1][W-1]
        # 4 borders
        img_aug[0:b-1][a:W_aug-1-a] = img[0][:]
        img_aug[b:H_aug-1-b][0:a-1] = img[:][0]
        img_aug[H:H_aug-1][a:W_aug-1-a] = img[H-1][:]
        img_aug[b:H_aug-1-b][W:W_aug-1] = img[:][W-1]

    # Initialization of final image with all 0
    img_median = [[0 for x in range(W)] for y in range(H)]
    for i, j in range(H), range(W):

        # Center of window: img_aug[i+b][j+a]
        window = img_aug[i:i+n-1][j:j+m-1]

        flat_window = []
        for k in range(len(window)):
            flat_window = flat_window + window[k]

        # Get median of all elements in the window
        img_median[i][j] = statistics.median(flat_window)

    return img_median

img = cv2.imread("image.jpg")
B, G, R = cv2.split(img)
knl = np.ones((5,5),np.float32)
B_mb = medianBlur(B, knl, "ZERO")
G_mb = medianBlur(G, knl, "ZERO")
R_mb = medianBlur(R, knl, "ZERO")
img_merge = cv2.merge((B_mb, G_mb, R_mb))
cv2.imshow("median_blur_lenna", img_merge)
key = cv2.waitKey(0)
cv2.destroyAllWindows()