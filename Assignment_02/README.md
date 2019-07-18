## Convolutional Filtering

### Gaussian blur
slkdfklsf
qklsjdlfq


### Bilateral filter
A __bilateral filter__ is a __non linear__, __edge-preserving__ and __noise-reducing smoothing__ filter.  
It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution. Crucially, the weights depend not only on Euclidean distance of pixels, but also on the radiometric differences (e.g., range differences, such as color intensity, depth distance, etc.). This preserves sharp edges.  
双边滤波是采用加权平均的方法，用周边像素亮度值的加权平均代表某个像素的强度，所用的加权平均基于高斯分布。最重要的是，双边滤波的权重不仅考虑了像素的欧氏距离（如普通的高斯低通滤波，只考虑了位置对中心像素的影响），还考虑了像素范围域中的辐射差异（例如卷积核中像素与中心像素之间相似程度、颜色强度，深度距离等），在计算中心像素的时候同时考虑这两个权重。 

### Conclusion
Averaging blur: fastest, can't keep edges  
Gaussian blur: slow, keep edges well
Median blur: can remove salt-and-pepper noises
Bilateral blur: sloweset, keep edges best


* Reading
1. bilateral filter
https://blog.csdn.net/piaoxuezhong/article/details/78302920
2. feature descriptors  
i. compulsory  
    a. Hog https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    b. SURF https://www.vision.ee.ethz.ch/~surf/eccv06.pdf
ii. optional  
    a. BRISK http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.1343&rep=rep1&type=pdf
    b. Orb http://www.willowgarage.com/sites/default/files/orb_final.pdf
3. preview parts  
K-Means
* Coding
1. 2D convolution filtering
2. reading + pseudo code
```python
#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A, 
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding 
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like: 
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#       
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (chinese: neidian) while 
#          others "outlier" (chinese: waidian)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold 
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List
#
```

Classical Image Stitching  
follow the instructions shown in the slides  
inputs are two images  
ouput is a stitched image  
in 2-3 weeks