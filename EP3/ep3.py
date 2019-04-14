import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import sklearn.datasets

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def cost_function(h, y):
    N = y.shape[0]
    cost = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    cost *= (-1/N)

    return cost

def logistic_fit(X, y, w = None, batch_size = None, learning_rate = 1e-2, num_iterations = 1000, return_history = False):
    '''
    Função que encontra o vetor de pesos

    :param X: array 2D (N x d) contendo N amostras de dimensão d
    :param y: array 1D (N x 1) contendo os labels (+1 ou -1)
    :param w: array 1D (d + 1 x 1) correspondendo a um vetor de pesos incial
    :param batch_size: quantidade de pontos usados para fazer update no vetor de pesos
    :param learning_rate: parâmetro real do gradient descent
    :param num_iterations: quantidade de vezes que o conjunto X é percorrido
    :param return_history: booleano, controla output da função
    :return w_final: array 1D (d + 1 x 1) de pesos ao final das iterações
    '''

    N, d = X.shape
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    X_sample = X
    y_sample = np.reshape(y, (y.shape[0], 1))

    cost_history = {}
    
    if w == None:
        w = [np.random.uniform(-1, 1) for _ in range(d + 1)]
        w = np.reshape(w, (d + 1, 1))

    for i in range(num_iterations):
        # sample batch size parameter
        if batch_size:
            idx = np.random.choice(N, batch_size, replace = False)
            X = X_sample[idx, :]
            y = y_sample[idx, :]

        # compute gradient
        z = np.dot(X, w)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / N

        # append to cost history (return history parameter)
        cost_history[i] = cost_function(h, y)

        # update w
        w -= learning_rate * gradient

    if return_history:
        for k, v in cost_history.items():
            print("Iteration {}: {}".format(k, v))

    return w


def logistic_predict(X, w):
    '''
    Função de predição

    :param X: array 2D (N x d) contendo N amostras de dimensão d
    :param w: array 1D (d + 1 x 1) correspondendo a um vetor de pesos incial
    :return predictions: array 1D (N x 1) de predições
    '''
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    z = np.dot(X, w)
    predictions = sigmoid(z)
    
    return predictions

def generate_dataset(mean1, mean2, cov1, cov2, size1, size2):
    x1 = np.random.multivariate_normal(mean1, cov1, size1).T
    y1 = np.ones((x1.shape[1], 1))
    x2 = np.random.multivariate_normal(mean2, cov2, size2).T
    y2 = np.zeros((x2.shape[1], 1))
    
    X = np.concatenate((x1.T, x2.T), axis = 0)
    y = np.concatenate((y1, y2), axis = 0)

    return (X, y)


def plot_predictions(X, pred):
    for i in range(len(pred)):
        if pred[i] > 0.5:
            color = 'r'
        else:
            color = 'b'
        plt.plot(X.T[0][i], X.T[1][i], 'x', marker = '.', color = color)
    plt.show()
    

mean1 = (4, 2)
mean2 = (10, 2)
cov1 = [[2, 0], [0, 2]]

X, y = generate_dataset(mean1, mean2, cov1, cov1, 1000, 1000)

w = logistic_fit(X, y, learning_rate=0.1, num_iterations=10000, return_history=True)
pred = logistic_predict(X, w)

plot_predictions(X, pred)






