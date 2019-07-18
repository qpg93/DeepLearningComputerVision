import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow("Lenna", img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian Kernel Effect (blur)
g_img = cv2.GaussianBlur(img, (7,7), 5)
cv2.imshow("Gaussian_Blur_Lenna", g_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Image is blurrer because of bigger kernel, more remarkable effect of averaging
g_img = cv2.GaussianBlur(img, (17,17), 5)
cv2.imshow("Gaussian_Blur_Lenna", g_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Image is clearer because of smaller variance, gaussian blur image is sharpened, more impact of the center of the kernel
g_img = cv2.GaussianBlur(img, (7,7), 1)
cv2.imshow("Gaussian_Blur_Lenna", g_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian kernel with one dimension
kernel = cv2.getGaussianKernel(7, 5)
print(kernel)

# One-dimensional operation is fast
g1_img = cv2.GaussianBlur(img, (7,7), 5)
g2_img = cv2.sepFilter2D(img, -1, kernel, kernel) # Original image, depth, kernelX, kernelY
cv2.imshow('g1_blur_lenna', g1_img)
cv2.imshow('g2_blur_lenna', g2_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

########## Other Applications ##########
# 2nd derivative: laplacian (bilateral effect) => show edges / borders
kernel_lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 9]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel = kernel_lap)
cv2.imshow('laplacian_lenna', lap_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Application: sharpen image
"""
This is not correct. There are four 1 in the kernel and one -3 in the center. Although the four 1 will enhance edge effect, they also give the kernel the filtering effect which make the image blur
kernel_sharp = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 9]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel = kernel_sharp)
cv2.imshow('laplacian_lenna', lap_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Solution: take the inverse of kernel_lap, add the original image. In this way, the center pixel is enhanced, the effect is similar to the gaussian with small variance => therefore, both edge effect and resolution will be kept
kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel = kernel_sharp)
cv2.imshow('sharp_lenna', lap_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# More sharp edge effect, even the diagonal direction is considered
kernel_sharp = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel = kernel_sharp)
cv2.imshow('more_sharp_lenna', lap_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

########## Edge ##########
# Vertical axis
edgex = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], np.float32)
sharp_img = cv2.filter2D(img, -1, kernel=edgex)
cv2.imshow('edgex_lenna', sharp_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Horizontal axis
edgey = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], np.float32)
sharp_img = cv2.filter2D(img, -1, kernel=edgey)
cv2.imshow('edgey_lenna', sharp_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

########## Corner ##########
img = cv2.imread('board.jpg')
img = cv2.resize(img, (640, 480))
img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
print(img_gray)

# 2: block/window size
# 3: Sobel kernel size
# 0.05: empirically determined constant between 0.04 and 0.06
img_harris = cv2.cornerHarris(img_img_gray, 2, 3, 0.05)
cv2.imshow('img_harris', img_harris)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
print(img_harris)

# To display more clearly
# img_harris = cv2.dilate(img_harris, None)

thres = 0.05 * np.max(img_harris)
img[img_harris > thres] = [0, 0, 255]
cv2.imshow('img_harris', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

########## SIFT ##########
"""
1. Generate Scale-Space: DoG
    Use a series of gaussian kernel to create blur images on different scales, do subtraction one from another => get the contour/outline/edges
2. Scale-Space Extrema Detection
    Find maximums and minimums on different scales
3. Accurate Keypoint Localization
    Accurate localization of extrema
4. Elimination of Edge Response
5. Orientation Assignment
6. Keypoint Descriptor
"""
"""To use SIFT, uninstall all other opencv
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
"""

img = cv2.imread('lenna.png')
# Create SIFT class
sift = cv2.xfeatures2d.SIFT_create()
# Detect SIFT
kp = sift.detect(img, None)   # None for mask
# Compute SIFT descriptor
kp, des = sift.compute(img,kp)
print(des.shape)
img_sift= cv2.drawKeypoints(img,kp,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('lenna_sift.jpg', img_sift)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
