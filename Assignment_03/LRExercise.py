# Linear Regression

# Library
import numpy as np
import random

# Inferenc: get prediction
def inference(w, b, x):
    pred_y = w * x + b
    return pred_y

# Gradient: get dw & db
def gradient(pred_y, gt_y, x):
    """
    pred_y: predicted output calculated
    gt_y: ground truth (already known)
    """

    diff = pred_y - gt_y
    dw = diff * x    # dtheta = (pred_y - gt_y) * x
    db = diff        # theta_0 = pred_y - gt_y
    return dw, db

# Calculate gradient of each step
def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw, avg_db = 0, 0
    batch_size = len(batch_x_list)

    for i in range(batch_size):
        pred_y = inference(w, b, batch_x_list[i])
        dw, db = gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
        avg_dw += dw
        avg_db += db
    avg_dw /= batch_size
    avg_db /= batch_size
    w = w - lr * avg_dw
    b = b - lr * avg_db

    return w, b

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w, b = 0
    num_samples = len(x_list)
    for i in range(max_iter):
        # SGD: Stochastic Gradient Descent (applicable for batch_size = 1)
        ## Minibatch SGD if batch_size > 1
        ### Gradient Descent if batch_size = num_samples
        batch_idxs = np.random.choice(len(x_list), batch_size) # Select batchsize samples from 0 to numsamples
