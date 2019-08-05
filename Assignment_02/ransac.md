# Random Sample Consensus (RANSAC)

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
