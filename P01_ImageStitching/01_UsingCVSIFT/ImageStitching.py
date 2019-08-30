import numpy as np
import cv2

# coding: utf-8
import numpy as np
import cv2

left_img = cv2.imread("left.jpg")
left_img = cv2.resize(left_img, (600, 400))
right_img = cv2.imread("right.jpg")
right_img = cv2.resize(right_img, (600, 400))
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Set Hessian Threshold to 400 (Bigger the threshold, less the features that can be detected)
hessian = 300
surf = cv2.xfeatures2d.SIFT_create(hessian)
kp1, des1 = surf.detectAndCompute(left_gray, None)  # Look for key points and descriptors
kp2, des2 = surf.detectAndCompute(right_gray, None)

# kp1s = np.float32([kp.pt for kp in kp1])
# kp2s = np.float32([kp.pt for kp in kp2])

# Draw key points
img_with_drawKeyPoint_left = cv2.drawKeypoints(left_gray, kp1, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_left", img_with_drawKeyPoint_left)

img_with_drawKeyPoint_right = cv2.drawKeypoints(right_gray, kp2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_right", img_with_drawKeyPoint_right)

'''
BFMatcher: Brute-Force Matcher, which means trying all matches and find the best one

FlannBasedMatcher: Fast Library for Approximate Nearest Neighbors, which is not necessarily the best matche but much faster
Parameters can be adjusted to increase the accuracy or change the speed of algorithm
Reference: https://blog.csdn.net/claroja/article/details/83411108
'''
FLANN_INDEX_KDTREE = 0  # FLANN parameter setting

indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # Set index, 5 KD trees
searchParams = dict(checks=50)  # Set iteration
# FlannBasedMatcher: fastest feature match algorithm now (search for the approximate nearest neighbor)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # Set matcher

# Reference: https://docs.opencv.org/master/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89
'''
int queryIdx –>是测试图像的特征点描述符（descriptor）的下标，同时也是描述符对应特征点（keypoint)的下标。

int trainIdx –> 是样本图像的特征点描述符的下标，同样也是相应的特征点的下标。

int imgIdx –>当样本是多张图像的话有用。

float distance –>代表这一对匹配的特征点描述符（本质是向量）的欧氏距离，数值越小也就说明两个特征点越相像。
'''

matches = flann.knnMatch(des1, des2, k=2)  # Get matched key points

good = []
# Extract good feature points
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # Descriptor index of searching image
dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # Descriptor index of training/model image

# findHomography: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
# Homography: https://www.cnblogs.com/wangguchangqing/p/8287585.html
H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)  # Generate matrix of transformation

h1, w1 = left_gray.shape[:2]
h2, w2 = right_gray.shape[:2]
shift = np.array([[1.0, 0, w1], [0, 1.0, 0], [0, 0, 1.0]])
# Dot product / Scalar product
M = np.dot(shift, H[0])  # Get the mapping from left image to right image

# Perspective transformation, new image could contain completely 2 images
dst = cv2.warpPerspective(left_img, M, (w1+w2, max(h1, h2)))
cv2.imshow('left_img', dst)  # Display left image in the standard position
dst[0:h2, w1:w1+w2] = right_img  # Put 2nd image on the right
# cv2.imwrite('tiled.jpg',dst_corners)
cv2.imshow('total_img', dst)
cv2.imshow('leftgray', left_img)
cv2.imshow('rightgray', right_img)
cv2.waitKey(0)
cv2.destroyAllWindows()