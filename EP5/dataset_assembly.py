import numpy as np

def select_random_examples(x, y, i): 
    data = x[y == i]
    target = y[y == i]
    idxs = np.random.choice(data.shape[0], size = 500, replace = False)
    
    return data[idxs, :, :], target[idxs]


def flatten_examples(x):
    num_pixels = x.shape[1] * x.shape[2]
    x = x.reshape(x.shape[0], num_pixels)
    
    return x


def create_dataset(x_train, y_train, x_test, y_test):
    X_test = np.concatenate([x_test[y_test == i] for i in range(5)], axis = 0)
    Y_test = np.concatenate([y_test[y_test == i] for i in range(5)], axis = 0).astype('float32')

    X_train = []
    Y_train = []
    for i in range(5):
        x_train_i, y_train_i = select_random_examples(x_train, y_train, i)
        X_train.append(x_train_i)
        Y_train.append(y_train_i)

    X_train = np.concatenate(X_train, axis = 0)
    Y_train = np.concatenate(Y_train, axis = 0).astype('float32')

    # Normalize pixels to [0, 1] interval    
    X_train = np.divide(flatten_examples(X_train), 255)
    X_test = np.divide(flatten_examples(X_test), 255)

    return X_train, Y_train, X_test, Y_test