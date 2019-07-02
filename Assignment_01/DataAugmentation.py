'''
Date: 20190702
Author: Qing Peng

This script is for the data augmentation of images. It is based on traditional computer vision techniques, including image crop, color shift, rotation and perspective transform.
'''

import cv2
import random
import numpy as np

# Image crop
def crop(img):
    row = img.shape[0]
    col = img.shape[1]

    row_crop = round(row*random.uniform(0.7,1))
    col_crop = round(row*random.uniform(0.7,1))

    img_crop = img[0:row_crop, 0:col_crop]

    return img_crop

# Color shift
def color_shift(img):
    # Brightness
    B, G, R = cv2.split(img)
    
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
        
    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
        
    img_merge = cv2.merge((B, G, R)) # combine 3 channels
    
    return img_merge

# Rotation
def rotation(img):
    angle = random.randint(-90, 90)
    scale = round(random.uniform(0.5,1.5), 3)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, scale) # centering, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) # apply to the image
    return img_rotate

# Perspective Transform
def perspective_transform(img):
    row, col = img.shape[0], img.shape[1]
    height, width, channels = img.shape
    
    # Warp
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width-1-random_margin, width-1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width-1-random_margin, width-1)
    y3 = random.randint(height-1-random_margin, height-1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height-1-random_margin, height-1)
    
    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width-1-random_margin, width-1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width-1-random_margin, width-1)
    dy3 = random.randint(height-1-random_margin, height-1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height-1-random_margin, height-1)
    
    # Original points
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    # Target points
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    
    return img_warp

img = cv2.imread("mark.jpg")
for i in range(20):
    img_a = rotation(img)
    img_b = crop(img_a)
    img_c = perspective_transform(img_b)
    img_d = color_shift(img_c)
    cv2.imwrite("img/mark"+str(i)+".jpg", img_d)