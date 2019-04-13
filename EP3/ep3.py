import numpy as np
import random

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

    it = 0
    N, d = X.shape
    
    if w == None:
        #pass
        w = []
        w = [np.random.uniform(0.1, 5) for _ in range(d + 1)]
        w = np.reshape(w, (d + 1, 1))

    X = np.concatenate((np.ones((N, 1)), X), axis=1)

    print(X.shape)
    print(w.shape)

    while it <= num_iterations:
        z = np.dot(X, w)
        h = 1.0 / (1 + np.exp(-z))
        print(y.shape)
        gradient = np.dot(X.T, (h - y)) / N 
        w = w - learning_rate * gradient
        it += 1

    return w


def logistic_predict(X, w):
    '''
    Função de predição

    :param X: array 2D (N x d) contendo N amostras de dimensão d
    :param w: array 1D (d + 1 x 1) correspondendo a um vetor de pesos incial
    :return predictions: array 1D (N x 1) de predições
    '''
    
    z = np.dot(X, w)
    predictions = 1.0 / (1 + np.exp(-z))


    return predictions

def plot_2D(mean1, mean2, cov1, cov2, size1, size2):
    x1, y1 = np.random.multivariate_normal(mean1, cov1, size1).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, size2).T

    X = np.concatenate((x1.T, x2.T), axis = 0)
    y = np.concatenate((y1.T, y2.T), axis = 0)

    return (X, y)

mean1 = (4, 2)
mean2 = (10, 2)
cov1 = [[2, 0], [0, 2]]

X, y = plot_2D(mean1, mean2, cov1, cov1, 10000, 10000)

# size = len(X)
# y1 = np.ones((size, 1))
# y2 = [-1 for _ in range(size)]
# y2 = np.reshape(y2, (size, 1))

# y = np.concatenate((y1, y2), axis = 0)

w = logistic_fit(np.reshape(X, (X.shape[0], 1)), np.reshape(y, (y.shape[0], 1)))

# pred = logistic_predict(X, w)