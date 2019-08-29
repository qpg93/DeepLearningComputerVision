import numpy as np
import scipy as sp
import scipy.linalg as sl
import cv2
import pylab

def ransac(data, model, min_sample, max_iteration, error_threshold, inliner_num_threshold, debug=False, return_all=False):
    """
    Reference: http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
    Pseudo code: http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
    Inputs:
        data - Sample points
        model - Hypothetical model
        min_sample - Minimum number of data points required to estimated model parameters
        max_iteration - Maximum number of iteration allowed
        error_threshold - Threshold to determine data points that are fit well by model
        inliner_num_threshold - number of close data points required to assert that a model fits well to data
    Output:
        bestfit - model parameters which best fit the data (returns None if no good model is found)
    """

    iteration = 0
    best_fit = None
    best_error = np.inf # Set default value
    best_inlier_idxs = None

    while (iteration < max_iteration):
        maybe_idxs, test_idxs = random_partition(min_sample, data.shape[0]) # Get random index
        maybe_inliers = data[maybe_idxs, :] # Get data (Xi,Yi) of size(maybe_idxs) rows
        test_points = data[test_idxs, :] # Get data points of certain (Xi,Yi) rows

        maybe_model = model.fit(maybe_inliers) # Fit the model
        test_errors = model.get_error(test_points, maybe_model) # Calculate the error: SSE (Sum of Squared Errors)

        also_idxs = test_idxs[test_errors < error_threshold] # Recognize additional inliers
        also_inliers = data[also_idxs, :]

        if debug:
            print('test_errors.min()', test_errors.min())
            print('test_errors.max())', test_errors.max())
            print('numpy.mean(test_errors)', np.mean(test_errors))
            print('iteration %d:len(also_inliers) = %d' % (iteration, len(also_inliers)))

        if (len(also_inliers) > inliner_num_threshold - min_sample ): # Enough inliers so that a good model is found
            better_data = np.concatenate((maybe_inliers, also_inliers)) # Sample concatenation, combine all inliers
            better_model = model.fit(better_data)
            better_errors = model.get_error(better_data, better_model) # Error is calculated without considering outliers
            this_error = np.mean(better_errors) # Use mean error as the new error
            if this_error < best_error:
                best_fit = better_model
                best_error = this_error
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs)) # Update inliers
        
        iteration += 1
    
        if best_fit == None:
            raise ValueError("didn't error_threshold meet fit acceptance criteria")

        if return_all:
            return best_fit, {'inliers': best_inlier_idxs}

        else:
            return best_fit

def random_partition(min_sample, number_data):
    """
    return n random rows of data and the other len(data) - n rows
    """
    all_idxs = np.arange(number_data) # Get number_data index
    np.random.shuffle(all_idxs) # Shuffles all index
    idxs_1 = all_idxs[:min_sample] # min_sample datapoints to estimate model if no also_inliers
    idxs_2 = all_idxs[min_sample:]
    return idxs_1, idxs_2

class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    
    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 1st col Xi--> row Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 2nd col Yi--> row Yi
        x, resids, rank, s = sl.lstsq(A, B) # Compute least square solution to equation Ax=B
        # resids: Square of the 2-norm for each column in B-Ax
        return x  # Least-squares solution

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 1st col Xi--> row Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 2nd col Yi--> row Yi
        B_fit = sp.dot(A, model)
        error_per_point = np.sum((B - B_fit)**2, axis=1) # Sum squared error per row
        return error_per_point

def generate_data(num_samples, num_inputs, num_outputs, num_outliers):
    # Generate random datapoints with values between 0 and 20 (row vectors: 800 col, 1 row)
    A_exact = 20 * np.random.random((num_samples, num_inputs))
    perfect_fit = 60 * np.random.normal(size=(num_inputs, num_outputs)) # Generate randomly a slope
    B_exact = sp.dot(A_exact, perfect_fit) # B = AX

    # Add Gaussian noise, which Least Square Method can handle well
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # (700 * 1) row vector, stands for Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # (700 * 1) row vector, stands for Yi

    if True: # Add outliers
        all_idxs = np.arange(A_noisy.shape[0]) # Get index 0-799
        np.random.shuffle(all_idxs) # Shuffle all index
        outlier_idxs = all_idxs[:num_outliers] # 100 random outliers
        A_noisy[outlier_idxs] = 50 * np.random.random((num_outliers, num_inputs)) # Add noise and outliers Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(num_outliers, num_outputs)) # Add noise and outliers Yi
    
    return A_exact, B_exact, A_noisy, B_noisy, perfect_fit, outlier_idxs

def test():
    # Generate ideal data
    num_samples = 800
    num_inputs = 1
    num_outputs = 1
    num_outliers = 100
    A_exact, B_exact, A_noisy, B_noisy, perfect_fit, outlier_idxs = generate_data(num_samples, num_inputs, num_outputs, num_outliers)

    # Setup model
    all_data = np.hstack((A_noisy, B_noisy)) # Form([Xi,Yi]....) shape:(800,2) 800row, 2col
    input_columns = range(num_inputs) # 1st col x:0
    output_columns = [num_inputs + i for i in range(num_inputs)] # Last col y:1

    # Instantiate the class: Use Least Square to generate known model
    debug = True
    model = LinearLeastSquareModel(input_columns, output_columns, debug)

    linear_fit, resids, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    min_sample = 400
    max_iteration = 100
    error_threshold = 1e2
    inliner_num_threshold = 500
    
    ransac_fit, ransac_data = ransac(all_data, model, min_sample, max_iteration, error_threshold,\
         inliner_num_threshold, debug, return_all=True)

    if True:
        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # Array with rank=2

        if True:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # Scatter plot (Nuage de points)
            #pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'y.',
            #           label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k_maxIteration.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ =='__main__':
    test()