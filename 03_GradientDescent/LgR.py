# Logistic Regression with loss evolution plot

import numpy as np
import matplotlib.pyplot as plt

# Get prediction: ^y
def pred(w, b, x_list):
    x = np.array(x_list)
    z = w * x + b
    return 1 / (1 + np.exp(-z)) # Sigmoid

# Evaluation
def eval_loss(w, b, x_list, gt_y_list):
    pred_y = pred(w, b, x_list)
    gt_y = np.array(gt_y_list)
    n = len(gt_y_list)
    loss = -(np.sum(gt_y * np.log(pred_y) + (1-gt_y) * np.log(1-pred_y))) / n
    return loss

# Calculate gradient of each step
def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    batch_size = len(batch_gt_y_list)
    x = np.array(batch_x_list)
    gt_y = np.array(batch_gt_y_list)
    pred_y = pred(w, b, batch_x_list)
    dw = np.sum((pred_y - gt_y) * x) / batch_size
    db = np.sum((pred_y - gt_y)) / batch_size
    w -= lr * dw
    b -= lr * db
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
        batch_x = x_list[batch_idxs]
        batch_gt_y = gt_y_list[batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_gt_y, w, b, lr)
        loss = eval_loss(w, b, x_list, gt_y_list)
        evol_loss.append(loss)
        print("Iteration {0}   w:{1}   b:{2}   loss:{3}".format(i, w, b, loss))
    visu_loss(evol_loss)
    return None

def visu_loss(loss):
    plt.plot(range(len(loss), loss))
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
    x = 100 * np.random.rand(num_samples)
    gt_y = w * x + b + (np.random.rand(num_samples) - 0.5) * 2
    return x, gt_y

def run():
    x_list, gt_y_list = gen_sample_data()
    lr = 1e-4 # if learning rate too big => gradient exploration => loss displayed as NaN in pytorch
    max_iter = 200
    batch_size = 100
    train(x_list, gt_y_list, batch_size, lr, max_iter)
    return None

if __name__ == "__main__":
    run()
