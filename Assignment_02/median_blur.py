"""
20190719 by Qing Peng
Major update by using numpy to deal with padding

20190718 by Qing Peng
Upgrade the padding way part
Issues exist in line 97: no median for empty data

20190717 by Qing Peng
Realize the median blur image by using a sliding window on an image to find the median value within that crop
"""

import numpy as np
import cv2
import math

class imageBlur:
    def medianBlur(self, img, kernel=(3,3), padding_way='ZERO'):
        """
        img: list of list
        kernel: list of list
        padding_way: string
        """
        
        # Check padding_way
        padding_way = padding_way.upper()
        allowed = {'ZERO', 'REPLICA'}
        if padding_way not in allowed:
            raise ValueError('The `padding` argument must be `ZERO` or `REPLICA. Received: '+str(padding_way))

        # Initialize the transformed image
        output = np.zeros_like(img) # Same size as img, initialized with 0
        img_shape = img.shape

        # Padding
        # math.ceil(x) Return the ceiling of x as a float, the smallest integer value greater than or equal to x
        padding = ((math.ceil(kernel[0]/2),),(math.ceil(kernel[1]/2),)) # Tuple: rows, cols
        
        if len(img_shape) == 3:
            padding += ((0,),)
        
        # padding: ((2,), (2,), (0,))
  
        """padding[0][0] = 3, padding[1][-1] = 3
        >>> a = [[1,2], [3,4]]
        >>< np.lib.pad(a, ((3, 2), (2, 3)), 'minimum')
        array([[1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [3, 3, 3, 4, 3, 3, 3],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1]])
        """

        if padding_way == 'ZERO':
            img = np.lib.pad(img, padding, mode='constant', constant_values=0)
        if padding_way == 'REPLICA':
            img = np.lib.pad(img, padding, mode='edge')

        # Generate new image with median
        for j in range(img_shape[1]):
            for i in range(img_shape[0]):
                window = img[i:i+padding[0][0]*2, j:j+padding[1][0]*2]
                # window = img[i+1 : i+padding[0][0]*2+2, j+1 : j+padding[1][-1]*2+2]
                output[i, j, :] = np.median(window, axis=(0,1))

        return output

if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    lennaBlur = imageBlur()
    img_myblur = lennaBlur.medianBlur(img, kernel=(3,3), padding_way="REPLICA")
    img_cvblur = cv2.medianBlur(img, 3)

    cv2.imshow('lenna', img)
    cv2.imshow('my_median_blur_lenna', img_myblur)
    cv2.imshow('cv2_median_blur_lenna', img_cvblur)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

