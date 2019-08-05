## Image Filtering

### 1. Bilateral filter
A bilateral filter is a __non linear__, __edge-preserving__ and __noise-reducing smoothing__ filter.  
It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution. Crucially, the weights depend not only on Euclidean distance of pixels, but also on the radiometric differences (e.g., range differences, such as color intensity, depth distance, etc.). This preserves sharp edges.  

双边滤波是采用加权平均的方法，用周边像素亮度值的加权平均代表某个像素的强度，所用的加权平均基于高斯分布。最重要的是，双边滤波的权重不仅考虑了像素的欧氏距离（如普通的高斯低通滤波，只考虑了位置对中心像素的影响），还考虑了像素范围域中的辐射差异（例如卷积核中像素与中心像素之间相似程度、颜色强度，深度距离等），在计算中心像素的时候同时考虑这两个权重。  

### 2. Comparison of Image Smoothing Methods
| Filter         | Pros and Cons                     |
|----------------|-----------------------------------|
| Averaging blur | Fastest, can't keep edges         |
| Gaussian blur  | Slow, keep edges well             |
| Median blur    | Can remove salt-and-pepper noises |
| Bilateral blur | Slowest, keep edges best          |

### 3. Feature Descriptors

#### 3.1 Histograms of Oriented Gradient (HOG)
HOG is a feature descriptor used for __object detection__.
The technique counts occurences of gradient orientation in localized portions of an image. It is similar to Edge Orientation Histograms, SIFT and Shape Contexts, but HOG is computed on a dense gride of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.  

HOG描述器是在一个网格密集的大小统一的细胞单元（dense grid of uniformly spaced cells）上计算，而且为了提高性能，还采用了重叠的局部对比度归一化（overlapping local contrast normalization）技术。HOG描述器最重要的思想是：在一副图像中，局部目标的表象和形状（appearance and shape）能够被梯度或边缘的方向密度分布很好地描述。  

具体的实现方法是：首先将图像分成小的连通区域，我们把它叫细胞单元。然后采集细胞单元中各像素点的梯度的或边缘的方向直方图。最后把这些直方图组合起来就可以构成特征描述器。为了提高性能，我们还可以把这些局部直方图在图像的更大的范围内（我们把它叫区间或block）进行对比度归一化（contrast-normalized），所采用的方法是：先计算各直方图在这个区间（block）中的密度，然后根据这个密度对区间中的各个细胞单元做归一化。通过这个归一化后，能对光照变化和阴影获得更好的效果。

Since it operates on local cells, it is invariant to geometric and photometric transformations, except for object orientation. Such changes would only appear in larger spatial regions.  

The HOG descriptor is particularly suited for human detection in images.

Reference: [HOG for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

#### 3.2 Speeded Up Robust Features (SURF)
SURF is a patented local feature detector and descriptor. It can be used for object recognition, image registration, classification or 3D reconstruction.  
1. Interest point detection
2. Local neighborhood description
3. Matching

It is partly inspired by the scale-invariant feature transform (SIFT) descriptor. The standard version of SURF is __several times faster than SIFT__ and claimed by its authors to be __more robust__ against different image transformations than SIFT.  

To detect interest points, SURF uses an integer approximation of the __determinant of Hessian blob detector__, which can be computed with 3 integer operations using a precomputed integral image. Its feature descriptor is based on the sum of the Haar wavelet response around the point of interest. These can also be computed with the aid of the integral image.  

Reference: [SURF: Speeded Up Robust Features](https://www.vision.ee.ethz.ch/~surf/eccv06.pdf)

#### 3.3 BRISK & ORB
 a. BRISK http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.1343&rep=rep1&type=pdf
 b. Orb http://www.willowgarage.com/sites/default/files/orb_final.pdf

### 4. Ramdom Sample Consensus (RANSAC)
It is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates. It also can be interpreted as an outlier detection method.  
1. Select a random subset of the original data. Call it _hypothetical inliers_.
2. Fit a model to the set of hypothetical inliers.
3. All other data are then tested against the fitted model. Those points that fit the estimated model well, according to some model-specific _loss function_, are considered as part of the _consensus set_.
4. The estimated model is reasonably good if sufficiently many points have been classified as part of the consensus set.
5. The model may be improved by reestimating it using all members of the consensus set.
This procedure is repeated a fixed number of times, each time producing either a model which is rejected because too few points are part of the consensus set, or a refined model together with a corresponding consensus set size. In the latter case, we keep the refined model if its consensus set is larger than the previously saved model.  

__Pseudo code of general RANSAC__
```python
inputs:
    data:        a set of observations
    model:       a model to explain observed data points
    n:           minimum number of data points required to estimated model parameters
    k:           maximum number of iterations allowed in the algorithm
    t:           threshold value to determine data points that are fit well by model
    d:           number of close data points required to assert that a model fits well to data
outpus:
    bestFit:     model parameters which best fit the data (or null if no good model is found)
    
iterations = 0
bestFit = null
bestErr = something really large

while (iterations < k):
    maybeInliers = n randomly selected values from data
    maybeModel = model parameters fitted to maybeInliers
    alsoInliers = empty set
    
    for every point in data not in maybeInliers:
        if point fits maybeModel with an error < t:
            add point to alsoInliers
    
    if (the number of elements in alsoInliers > d):
        # this implies that we may have found a good model
        # now test how good it is
        betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr = a measure of how well betterModel fits these points
        if (thisErr < bestErr):
            bestFit = betterModel
            bestErr = thisErr
    
    iterations += 1

return bestFit
```

__Pseudo code of _ransacMatching___  
We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A, B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding B.m.  
However, it's obvious that the matching cannot be so perfect and the matching in our real world is like A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...  
The target is to find out the true matching within this messy.
```python
def ransacMatching(A, B):
    # A & B: List of List
    iterations = 0
    k = max number of iterations
    
    bestFit = null
    bestErr = something really large
    
    while (iterations < k):
        maybePairA = 4 randomly selected items from A
        maybePairB = 4 corresponding items from B (with same index)
        maybeModel = model parameters fitted to maybePairA and maybePairB (use tools e.g. linear regression)
        alsoPairA = empty
        alsoPairB = empty
        
        for every pairA in data not in maybePairA: 
            for every pairB in data not in maybePairB: 
                if pairA and pairB fit maybeModel with an error < t:
                    add pairA to alsoPairA
                    add pairB to alsoPairB
        
        if (number of elements in alsoPairA > d):
            # this implies that we may have found a good model
            # now test how good it is
            betterModel = model parameters fitted to all points in maybePair and alsoPair
            thisErr = a measure of how well betterModel fits these points (use tools e.g. mean square error)
            if (thisErr < bestErr):
                bestFit = betterModel
                bestErr = thisErr
        
        iterations += 1

    # use the bestModel to find the A-matched B list
    matchedA = A
    matchedB = bestFit(A)
    
    return matchedA, matchedB
```

Classical Image Stitching  
follow the instructions shown in the slides  
inputs are two images  
ouput is a stitched image  
in 2-3 weeks
