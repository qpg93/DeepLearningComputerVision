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
    :return: tuple ((x1, x2), y) of sample data
    '''
    trainX1 = np.array(100 * np.random.random_sample(size))
    trainX2 = np.array(100 * np.random.random_sample(size))

    def random_y(x1, x2):
        boundry = x1 * 0.6 - 6 # y=2x-6
        return 1 if x2>boundry else 0

    y = [random_y(trainX1[i], trainX2[i]) for i in range(size)]
    trainY = np.array(y, dtype=np.float)
    return (trainX1, trainX2), trainY

def sigmoid_model(X, Theta):
    '''
    :param X: (x1, x2) of sample data
    :param Theta: (theta0, theta1, theta2) for model z = theta0 * x1 + theta1 * x2 + theta2
                  and z for model 1 / (1 + np.exp(-z))
    :return: y of model
    '''
    Xmatrix = np.vstack((X, (np.ones_like(X[0]))))
    z = Theta @ Xmatrix
    Ypred = 1 / (1 + np.exp(-z)) # add Sigmoid function to Linear Regression model
    return Ypred

def cal_loss(Ygt, Ypred):
    '''
    :param Ygt: ground truth y of sample data
    :param Ypred: y calculated by the model
    :return: loss of model
    '''
    loss = - np.sum(Ygt * np.log(Ypred) + (1 - Ygt) * np.log(1 - Ypred)) / len(Ygt)
    return loss

def gradient_descent(X, Ygt, Ypred, Theta, lr):
    '''
    :param X: (x1, x2) of sample data
    :param Ygt: ground truth y of sample data
    :param Ypred: calculated output of the model
    :param Theta: (theta0, theta1, theta2) for current model
    :param lr: learning rate of training
    :return: (theta0, theta1, theta2) for next model
    '''
    num_samples = len(Ypred)
    theta0, theta1, theta2 = Theta

    dtheta0 = np.sum((Ypred - Ygt) * X[0]) / num_samples
    dtheta1 = np.sum((Ypred - Ygt) * X[1]) / num_samples
    dtheta2 = np.sum(Ypred - Ygt) / num_samples

    theta0 -= lr * dtheta0
    theta1 -= lr * dtheta1
    theta2 -= lr * dtheta2
    Theta = np.array((theta0, theta1, theta2))
    return Theta

def train(X, Ygt, training_epoch, learning_rate):
    '''
    :param X: x of sample data
    :param Y: ground truth y of sample data
    :training_epoch: number of iteration
    :learning_rate: learning rate of training
    :return: (theta0, theta1) for final model and all losses
    '''
    Theta = np.array(np.random.random_sample((3,)))
    cost = []
    for i in range(training_epoch):
        Ypred = sigmoid_model(X, Theta)
        Theta = gradient_descent(X, Ygt, Ypred, Theta, learning_rate)
        loss = cal_loss(Ygt, Ypred)
        cost.append(loss)
        print("Iteration{}   theta0:{}   theta1:{}   theta2:{}   loss:{}".format(i, Theta[0], Theta[1], Theta[2], loss))
    return Theta, cost

def draw(X, Ygt, Theta, cost):
    training_epoch = len(cost)

    fig, [plt1, plt2] = plt.subplots(2, 1)
    fig.suptitle("Logistic Regression")

    plt1.set_xlabel("X1")
    plt1.set_ylabel("X2")
    pos_samples = (Ygt==1)
    neg_samples = (pos_samples==False)
    pos = X[0][pos_samples], X[1][pos_samples]
    neg = X[0][neg_samples], X[1][neg_samples]

    plt1.scatter(pos[0], pos[1], label="Positive", color="green")
    plt1.scatter(neg[0], neg[1], label="Negative", color="red")
    plt1.legend()

    theta0, theta1, theta2 = Theta

    xs1 = np.linspace(0, 100, 100)
    xs2 = []

    if theta1 != 0:
        xs2 = [-(theta0 * x + theta2) / theta1 for x in xs1] # transform from theta0 x0 + theta1 x1 + theta2 = 0
    plt1.plot(xs1, xs2, label="Hypothesis", color='blue')

    plt2.plot(range(training_epoch), cost)
    plt2.set_xlabel("Training epoch")
    plt2.set_ylabel("Loss")

    plt.show()

def run():
    training_epoch = 200
    learning_rate = 1e-3
    sample_size = 300
    trainX, trainY = gen_sample_data(sample_size)
    Theta, cost = train(trainX, trainY, training_epoch, learning_rate)
    draw(trainX, trainY, Theta, cost)

if __name__ == "__main__":
    run()