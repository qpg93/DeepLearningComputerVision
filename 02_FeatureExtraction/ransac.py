import cv2
import numpy as np
import matplotlib
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n_minSample, k_maxIteration, t_errorThreshold, d, debug=False, return_all=False):
    """
    Reference: http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
    Pseudo code: http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
    Inputs:
        data - Sample points
        model - Hypothetical model
        n_minSample - Minimum number of data points required to estimated model parameters
        k_maxIteration - Maximum number of iteration allowed
        t_errorThreshold - Threshold to determine data points that are fit well by model
        d - number of close data points required to assert that a model fits well to data
    Output:
        bestfit - model parameters which best fit the data (returns nil if no good model is found)
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # Set default value
    best_inlier_idxs = None
    while iterations < k_maxIteration:
        maybe_idxs, test_idxs = random_partition(n_minSample, data.shape[0]) # Get random index
        maybe_inliers = data[maybe_idxs, :]  # Get data (Xi,Yi) of size(maybe_idxs) rows
        test_points = data[test_idxs]  # Get data points of certain (Xi,Yi) rows
        maybemodel = model.fit(maybe_inliers)  # Fit the model
        test_err = model.get_error(test_points, maybemodel)  # Calculate the error: SSE (Sum of Squared Errors)
        also_idxs = test_idxs[test_err < t_errorThreshold]
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        if len(also_inliers > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # Sample concatenation, combine all inliers
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel) # Error is calculated without considering outliers
            thiserr = np.mean(better_errs)  # Use mean error as the new error
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # Update inliers
        iterations += 1
        if bestfit is None:
            raise ValueError("did't_errorThreshold meet fit acceptance criteria")
        if return_all:
            return bestfit, {'inliers': best_inlier_idxs}
        else:
            return bestfit


def random_partition(n_minSample, n_data):
    """
    return n random rows of data and the other len(data) - n rows
    """
    all_idxs = np.arange(n_data)  # Get n_data index
    np.random.shuffle(all_idxs)  # Shuffles all index
    idxs1 = all_idxs[:n_minSample] # n_minSalple datapoints to estimate model if no also_inliers
    idxs2 = all_idxs[n_minSample:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # Least Squares Method, for RANSAC's input
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 1st col Xi--> row Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 2nd col Yi--> row Yi
        x, resids, rank, s = sl.lstsq(A, B)  # Compute least squares solution to equation Ax=B
        # resids: Square of the 2-norm for each column in B-Ax
        return x  # Least-squares solution

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 1st col Xi--> row Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 2nd Yi--> row Yi
        B_fit = sp.dot(A, model)  # Compute Y, B_fit = model.k_maxIteration*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def test():
    # Generate ideal data
    n_samples = 700 
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # Generate randomly 700 datapoints with values between 0 and 20 (row vectors: 700 col, 1 row)
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # Generate randomly a slope
    B_exact = sp.dot(A_exact, perfect_fit)  # B = AX, X=model

    # Add Gaussian noise, which Least Square Method can handle well
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # (700 * 1) row vector, stands for Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # (700 * 1) row vector, stands for Yi

    if 1:
        # 添加"局外点"
        n_outliers = 50
        all_idxs = np.arange(A_noisy.shape[0])  # Get index 0-699
        np.random.shuffle(all_idxs)  # Shuffle all_idxs
        outlier_idxs = all_idxs[:n_outliers]  # 50 random outliers
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # Add noise and outliers Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # Add noise and outliers Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # Form([Xi,Yi]....) shape:(700,2) 700row, 2col
    input_columns = range(n_inputs)  # 1st col x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # Last col y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # Instantiate the class: Use Least Square to generate known model

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC
    ransac_fit, ransac_data = ransac(all_data, model, 10, 1000, 1e2, 500, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # Array with rank=2

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # Scatter plot (Nuage de points)
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
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


if __name__=='__main__':
      test()
