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
    train_Y = 5 * train_X + 6 + np.random.randn(size) * 0.2 # y=5x+6+noise
    return train_X, train_Y

def lr_model(X, Theta):
    '''
    :param X: x of sample data
    :param Theta: (theta_0, theta_1) of model y = theta_0 + theta_1 * x
    :return: y of model
    '''
    Xmatrix = np.vstack(np.ones_like(X), X) # X0 = 1, X1 = X
    Ypred = Theta @ Xmatrix.transpose()
    return Ypred

def cal_cost(Ygt, Ypred):
    '''
    :param Y: y of sample data
    :param Ypred: y calculated by the model
    :return: loss of model
    '''
    loss = 0.5 * np.sum((Ypred - Ygt)**2) / len(Ygt)
    return loss

def gradient_descent(X, Y, Ypred, Theta, lr):
    '''
    https://github.com/zhangshuai365/assignment/blob/master/assignment3/q1_week3_LinearRegression.py
    '''
