# Image stitching with OpenCV: more than 2 images together into a panoramic image

# python stitch.py --image stitch_input --output stitch_output\output.jpg
# Reference: https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
from imutils import paths
import imutils
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, help="path to the input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, help="path to the output image")
args = vars(ap.parse_args())

# Grab the paths to the input images and initialize the image list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# Loop over the image paths, load each one, and add them to images of stitch list
for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    images.append(img)

# Initialize OpenCV's image stitcher object and then perform the image stitching
print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# If status is 0, then OpenCV sucessfully performed image stitching
if status == 0:
    # Write the output stitched image to disk
    cv2.imwrite(args["output"], stitched)
    print("[INFO] image stitching succeeded ({})".format(status))
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
# Otherwise the stitching failed, likely due to not enough keypoints being detected
else:
    print("[INFO] image stitching failed ({})".format(status))

# Crop only the region of panorama in