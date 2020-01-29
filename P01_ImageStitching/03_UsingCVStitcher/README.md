# Classical Image Stitching
| Inputs     | Output             |
|:-----------|:-------------------|
| Two images | One stitched image |  

__Pipeline__
1. Find ___SIFT___ feature points in each image
1. Use ___RANSAC___ to find keypoint matches
1. Use ___homography matrix___ to get transferring into
1. Merge two images