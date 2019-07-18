"""
20190718 by Qing Peng
Upgrade the padding way part
Issues exist in line 97: no median for empty data

20190717 by Qing Peng
Realize the median blur image by using a sliding window on an image to find the median value within that crop
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
    m, n = len(kernel[0]), len(kernel)

    #if m%2 == 0 or n%2 == 0:
    #    return None

    # Size of augmented image with padding = W + 2*(m-1)/2, H + 2*(n-1)/2   
    # Padding
    a = int((m - 1) / 2)
    b = int((n - 1) / 2)
    # Augmented W and H
    Wa = W + 2*a
    Ha = H + 2*b

    # Initialization of the augmented padding image with all 0
    img_aug = [[0 for x in range(Wa)] for y in range(Ha)]
    # Original image in the center
    for i in range(H):
        for j in range(W):
            img_aug[i+b][j+a] = img[i][j]

    if padding_way == "ZERO":
        pass
    elif padding_way == "REPLICA":
        # 4 corners
        for i in range(b):
            for j in range(a):
                img_aug[i][j] = img[0][0]
                img_aug[i][j+W+a] = img[0][W-1]
                img_aug[i+H+b][j] = img[H-1][0]
                img_aug[i+H+b][j+W+a] = img[H-1][W-1]

        # 4 borders
        for i in range(H):
            for j in range(a):
                img_aug[i+b][j] = img[i][0]
                img_aug[i+b][j+W-1] = img[i][W-1]
        for i in range(b):
            for j in range(W):
                img_aug[i][j+a] = img[0][j]
                img_aug[i+H-1][j+a] = img[H-1][j]

    # Initialization of final image with all 0
    img_median = [[0 for x in range(W)] for y in range(H)]
    for i in range(H):
        for j in range(W):
            flat_window = []
            # Center of window: img_aug[i+b][j+a]
            window = img_aug[i:i+n-1][j:j+m-1]

            for k in range(len(window)):
                flat_window = flat_window + window[k]

            # Get median of all elements in the window
            img_median[i][j] = statistics.median(flat_window)

    return img_median

img = cv2.imread("lenna.png")
cv2.imshow("lenna", img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

img_cvMB = cv2.medianBlur(img, 5)
cv2.imshow("cv2_median_blur_lenna", img_cvMB)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

B, G, R = cv2.split(img)
knl = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]] # Actually only the kernel size matters
B_mb = medianBlur(B, knl, "REPLICA")
G_mb = medianBlur(G, knl, "REPLICA")
R_mb = medianBlur(R, knl, "REPLICA")
print(B_mb)
img_myMB = cv2.merge((B_mb, G_mb, R_mb))
cv2.imshow("my_median_blur_lenna", img_myMB)
key = cv2.waitKey(0)
cv2.destroyAllWindows()