# coding: utf-8

import numpy as np
import cv2
import os

def imShowWithWaitKey(img):
    cv2.imshow('Image', img)
    if cv2.waitKey() == 27: # ASCII code of ESC
        cv2.destroyAllWindows()

def getPoints(img, hessianThreshold, savePath = None):
    '''
    return feature points and descriptions
    '''
    # Create SURF detector
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold = hessianThreshold)

    # Find SURF feature points and describor
    fpts, desc = detector.detectAndCompute(img, None)
    
    # Draw feature points
    img_fp = cv2.drawKeypoints(img, fpts, outImage = img.copy(), flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    if savePath:
        imShowWithWaitKey(img_fp)
        cv2.imwrite(savePath, img_fp)

    return fpts, desc

def keyPointsMatch(img_1, fpts_1, desc_1, img_2, fpts_2, desc_2):
    '''
    return img_3 with matched points drawn on
    '''
    # Define a point matcher
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)

    # Match "nearest points" by Manhanttan Distance
    matches = matcher.match(desc_1, desc_2)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)[:20]

    # Draw first 20 matches
    img_3 = cv2.drawMatches(img_1, fpts_1, img_2, fpts_2, matches, outImg = np.array([0]), flags = 2)

    # Get points' localization
    fp_matched_1 = [fpts_1[i.queryIdx].pt for i in matches]
    fp_matched_2 = [fpts_2[i.trainIdx].pt for i in matches]

    return img_3, np.array(fp_matched_1), np.array(fp_matched_2)

def imgStitch(img_1, fp_matched_1, img_2, fp_matched_2, method = 1):
    '''
    return stitched image
    '''
    # Find Homography matrix
    homo, _ = cv2.findHomography(fp_matched_1, fp_matched_2, cv2.RANSAC)

    # homo/np.linalg.inv(homo)
    img_4 = cv2.warpPerspective(img_2, homo, (img_1.shape[1] + img_2.shape[1], img_1.shape[0]))

    sum_row = img_4[:,:,0].sum(axis = 0)
    endRight = np.where(sum_row > 0)[0][-1] # End the edge on the right

    if method:
        # Add weight
        right = img_1.shape[1] # Right edge
        left = np.where(sum_row > 0)[0][0] # Left edge

        # Give weight according to the distances
        for col in range(right):
            if col <= left:
                img_4[:,col,:] = img_1[:,col,:]
            else:
                a = img_1[:,col,:]
                b = img_4[:,col,:]
                b = np.where(b == 0, a, b) # Fill empty areas with a
                w1 = (col - left)/(right - left)
                w2 = (right - col)/(right - left)
                img_4[:,col,:] = w2 * a + w1 * b
                # print(w1 * a + w2 * b)
        img_4 = img_4[:, :endRight, :] # Cut the black borders on the right

    else:
        # Superpose
        img_4[0:img_1.shape[0], 0:img_1.shape[1]] = img_1

    img_4 = img_4[:, :endRight, :] # Cut the black borders on the right

    return img_4

if __name__ == '__main__':
    path_1 = os.path.join(os.path.dirname(__file__), "input", "1.jpg")
    path_2 = os.path.join(os.path.dirname(__file__), "input", "2.jpg")
    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)

    fpts_1, desc_1 = getPoints(img_1, hessianThreshold = 1000, savePath = os.path.join(os.path.dirname(__file__), "output", "fp1.jpg"))
    fpts_2, desc_2 = getPoints(img_2, hessianThreshold = 1000, savePath = os.path.join(os.path.dirname(__file__), "output", "fp2.jpg"))

    img_3, fp_matched_1, fp_matched_2 = keyPointsMatch(img_2, fpts_2, desc_2, img_1, fpts_1, desc_1)
    imShowWithWaitKey(img_3)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "output", "fpMatches.jpg"), img_3)

    img_4 = imgStitch(img_1, fp_matched_1, img_2, fp_matched_2, method = 1)

    imShowWithWaitKey(img_4)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "output", "stitchedImg.jpg"), img_4)