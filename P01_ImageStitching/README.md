# Classical Image Stitching
|Inputs|Two images|
|Output|One stitched image|
__Pipeline__
1. Find feature points in each image
2. Use ___RANSAC___ to find keypoint matches
3. Use ___homography matrix___ to get transferring into
4. Merge two images