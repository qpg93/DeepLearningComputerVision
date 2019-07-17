"""
20190716 by Qing Peng
Add arguments of functions and argparser in order to be adapted to more realistic use

20190713 by Qing Peng
Update function encapsulation for pratical industrial application by using class

20190702 by Qing Peng
This script is for the data augmentation of images. It is based on traditional computer vision techniques, including image crop, color shift, rotation and perspective transform.
"""


import numpy as np
import random
import cv2
import argparse # to write command line interface
import os # to use operating system dependent functionality

# Image crop
def crop(img, scale):
    """
    img: matrix of image
    scale: compare to the original image size, value from 0 to 1
    """

    row = img.shape[0]
    col = img.shape[1]

    # # At least some percentage of the original image
    row_crop = round(row*random.uniform(scale,1))
    col_crop = round(row*random.uniform(scale,1))

    img_crop = img[0:row_crop, 0:col_crop]

    return img_crop

# Color shift
def color_shift(img, shift=50):
    """
    img: matrix of image
    shift: shift value of pixel
    """

    # Brightness
    B, G, R = cv2.split(img)
    
    b_rand = random.randint(-abs(shift), abs(shift))
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
        
    g_rand = random.randint(-abs(shift), abs(shift))
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

    r_rand = random.randint(-abs(shift), abs(shift))
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
def rotation(img, angle=30, zoom=1):
    """
    img: matrix of image
    angle: angle to rotate
    scale: compare to the original image size, zoom in or out
    """
    rows, cols = img.shape[:2]
    scale = zoom
    M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, scale) # centering, angle, scale
    img_rotate = cv2.warpAffine(img, M, (cols, rows)) # apply to the image
    return img_rotate

# Perspective Transform
def perspective_transform(img, margin=30):
    """
    img: matrix of image
    margin: random value for generating coordinates
    """

    row, col = img.shape[:2]
    height, width, channels = img.shape
    
    # Warp
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

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_input", type=str, help="Path of input image")
    parser.add_argument("--img_output", type=str, help="Path of output image")
    parser.add_argument("--scale", type=float, help="Size after cropping, value from 0 to 1")
    parser.add_argument("--shift", type=int, help="Shift value of pixel")
    parser.add_argument("--angle", type=float, help="Angle to rotate")
    parser.add_argument("--zoom", type=float, help="Zooming size, value from 0 to 1")
    parser.add_argument("--margin", type=int, help="Value for generating coordinates")
    return parser.parse_args()

if __name__=="__main__":
    parse = argparser()
    img_dir = parse.img_input
    out_dir = parse.img_output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) # if the path doesn't exist, create one
    if os.path.exists(img_dir):
        # if the input image exists, do data augmentation after getting arguments
        for file in os.listdir(img_dir):
            if file.endswith(("jpg","png")):
                img = cv2.imread(os.path.join(img_dir, file))
                if parse.scale:
                    img = crop(img, parse.scale)
                if parse.angle and parse.zoom:
                    img = rotation(img, parse.angle, parse.zoom)
                if parse.shift:
                    img = color_shift(img, parse.shift)
                if parse.margin:
                    img = perspective_transform(img, parse.random_margin)
                cv2.imwrite(os.path.join(out_dir, file), img)