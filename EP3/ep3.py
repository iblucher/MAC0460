import numpy as np

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
    N = X.shape[0]
    
    if w == None:
        # initialize w randomly
        pass

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
    
    z = np.dot(X, w)
    predictions = 1.0 / (1 + np.exp(-z))


    return predictions

