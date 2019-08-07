# Linear Regression with loss evolution plot

import numpy as np
import matplotlib.pyplot as plt

# Get prediction: ^y
def pred(w, b, x):
    return w * x + b

# Evaluation
def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = 0.0
    n = len(gt_y_list)
    for i in range(n):
        avg_loss += (w * x_list[i] + b - gt_y_list[i]) ** 2
    avg_loss /= n
    return avg_loss

# Gradient: get dw and db
def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return dw, db

# Calculate gradient of each step
def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw, avg_db = 0, 0
    batch_size = len(batch_gt_y_list)
    for i in range(batch_size):
        pred_y = pred(w, b, batch_x_list[i])
        dw, db = gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
        avg_dw += dw
        avg_db += db
    avg_dw /= batch_size
    avg_db /= batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w, b = 0, 0
    num_samples = len(gt_y_list)
    evol_loss = []
    for i in range(max_iter):
        # SGD: Stochastic Gradient Descent (applicable for batch_size = 1)
        ## Minibatch SGD (if batch_size > 1)
        ### Gradient Descent (if batch_size == num_samples)
        batch_idxs = np.random.choice(num_samples, batch_size) # Generate a random sample from a given 1-D array
        batch_x = [x_list[j] for j in batch_idxs]
        batch_gt_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_gt_y, w, b, lr)
        loss = eval_loss(w, b, x_list, gt_y_list)
        evol_loss.append(loss)
        # print("w:{0}, b:{1}".format(w, b))
        # print("loss:{0}".format(loss))
    plt.plot(range(max_iter), evol_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()
    return None

# Generate sample data
def gen_sample_data():
    w = np.random.randint(0, 10) + np.random.rand()
    b = np.random.randint(0, 10) + np.random.rand()
    num_samples = 1000
    x_list, gt_y_list = [], []
    for i in range(num_samples):
        x = np.random.randint(0, 100) * np.random.rand()
        gt_y = w * x + b + np.random.rand() * np.random.randint(-1, 1)
        x_list.append(x)
        gt_y_list.append(gt_y)
    return x_list, gt_y_list

def run():
    x_list, gt_y_list = gen_sample_data()
    lr = 1e-3 # if learning rate too big => gradient exploration => loss displayed as NaN in pytorch
    max_iter = 2000
    batch_size = 100
    train(x_list, gt_y_list, batch_size, lr, max_iter)
    return None

if __name__ == "__main__":
    run()
