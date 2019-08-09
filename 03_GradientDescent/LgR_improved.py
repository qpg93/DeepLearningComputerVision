# -*- coding: utf-8 -*-

'''
step1:   generate sample data
step2:   training
step2.1: set up a hypothesis
step2.2: calculate the predict result according to the hypothesis
step2.3: calculate the loss (for visulization)
step2.4: calculate the gradient and descent it, generate a new model/hypothesis
step2.5: repeat from 2.2, till the epoch reaches threshold
step3:   show the result
'''

import numpy as np
import matplotlib.pyplot as plt

def gen_sample_data(size):
    '''
    :param size: number of sample data
    :return: tuple of sample data
    '''
    train_X = np.linspace(-1, 1, size)
    train_Y = 5 * train_X - 6 + np.random.randn(size) * 0.2 # y=5x+6+noise
    return train_X, train_Y

def lr_model(X, Theta):
    '''
    :param X: x of sample data
    :param Theta: (theta_0, theta_1) of model y = theta_0 + theta_1 * x
    :return: y of model
    '''
    Xmatrix = np.vstack((np.ones_like(X), X)) # X0 = 1, X1 = X
    Ypred = Theta @ Xmatrix
    return Ypred

def cal_loss(Ygt, Ypred):
    '''
    :param Ygt: ground truth y of sample data
    :param Ypred: y calculated by the model
    :return: loss of model
    '''
    loss = 0.5 * np.sum((Ypred - Ygt)**2) / len(Ygt)
    return loss

def gradient_descent(X, Ygt, Ypred, Theta, lr):
    '''
    :param X: x of sample data
    :param Ygt: ground truth y of sample data
    :param Ypred: calculated output of the model
    :param Theta: (theta0, theta1) for current model
    :param lr: learning rate of training
    :return: (theta0, theta1) for next model
    '''
    num_samples = len(Ypred)
    theta0, theta1 = Theta

    dtheta0 = np.sum(Ypred - Ygt) / num_samples
    dtheta1 = np.sum((Ypred - Ygt) * X) / num_samples

    theta0 -= lr * dtheta0
    theta1 -= lr * dtheta1
    Theta = (theta0, theta1)
    return Theta

def train(X, Ygt, training_epoch, learning_rate):
    '''
    :param X: x of sample data
    :param Y: ground truth y of sample data
    :training_epoch: number of iteration
    :learning_rate: learning rate of training
    :return: (theta0, theta1) for final model and all losses
    '''
    Theta = np.random.random_sample((2,))
    cost = []
    for i in range(training_epoch):
        Ypred = lr_model(X, Theta)
        Theta = gradient_descent(X, Ygt, Ypred, Theta, learning_rate)
        loss = cal_loss(Ygt, Ypred)
        cost.append(loss)
        print("Iteration {0}   theta1:{1}   theta0:{2}   loss:{3}".format(i, Theta[1], Theta[0], loss))
    return Theta, cost

def draw(X, Ygt, Theta, cost):
    training_epoch = len(cost)

    fig, [plt1, plt2] = plt.subplots(2, 1)
    fig.suptitle("Linear Regression")

    plt1.set_xlabel("X")
    plt1.set_ylabel("Ground truth Y")
    plt1.plot(X, Ygt, "b", label="Samples")

    x = np.linspace(-1, 1, len(X))
    ypred = Theta[0] + Theta[1] * x
    plt1.plot(x, ypred, "r", label="Hypothesis")
    plt1.legend()

    plt2.plot(range(training_epoch), cost)
    plt2.set_xlabel("Training epoch")
    plt2.set_ylabel("Loss")

    plt.show()

def run():
    training_epoch = 200
    learning_rate = 1e-1
    sample_size = 1000
    trainX, trainY = gen_sample_data(sample_size)
    Theta, cost = train(trainX, trainY, training_epoch, learning_rate)
    draw(trainX, trainY, Theta, cost)

if __name__ == "__main__":
    run()