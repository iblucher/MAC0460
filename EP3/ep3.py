import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

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
        w = [np.random.uniform(-10, 10) for _ in range(d + 1)]
        w = np.reshape(w, (d + 1, 1))
        """OI, ISA
        alterei os parâmetros do w pra ver se ajudava, mas não deu em muita coisa. 
        """

    X = np.concatenate((np.ones((N, 1)), X), axis=1)

    while it <= num_iterations:
        z = np.dot(X, w)
        h = 1.0 / (1 + np.exp(-z))
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
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    z = np.dot(X, w)
    predictions = 1.0 / (1 + np.exp(-z))
    
    return predictions

def plot_2D(mean1, mean2, cov1, cov2, size1, size2):
    x1 = np.random.multivariate_normal(mean1, cov1, size1).T
    y1 = np.ones((x1.shape[1], 1))
    x2 = np.random.multivariate_normal(mean2, cov2, size2).T
    y2 = np.negative(y1)
    
    X = np.concatenate((x1.T, x2.T), axis = 0)
    y = np.concatenate((y1, y2), axis = 0)

    return (X, y)

""" OI, ISA
Essa função tá aaabsurdamente lenta por ficar plotando um ponto de 
cada vez. Mas não consegui fazer plotar com um array de cores, como
tá na linha 81. tô com sono, send help
"""
def plot_it(X, pred):
    for i in range(len(pred)):
        if pred[i] > 0.5:
            color = 'r'
        else:
            color = 'b'
        plt.plot(X.T[0][i], X.T[1][i], 'x', marker = '.', color = color)
    plt.show()
    
    #plt.plot(X.T[0], X.T[1], 'x', marker = '.', color = colors)

mean1 = (4, 2)
mean2 = (10, 2)
cov1 = [[2, 0], [0, 2]]

X, y = plot_2D(mean1, mean2, cov1, cov1, 1000, 1000)

w = logistic_fit(X, y)
pred = logistic_predict(X, w)

print(pred)

plot_it(X, pred)


""" OI, ISA
https://www.youtube.com/watch?v=sGF6bOi1NfA
"""

#print()

# plt.plot(data1[0], data1[1], 'x', marker = '.')
# plt.plot(data2[0], data2[1], 'x', marker = '.')

# plt.axis('equal')
# plt.show()





