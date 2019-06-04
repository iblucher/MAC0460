import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def select_classes_0_to_4(x_train, y_train, x_test, y_test):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(5):
        x_train_i = x_train[y_train == i]
        y_train_i = y_train[y_train == i]
        x_test_i = x_test[y_test == i]
        y_test_i = y_test[y_test == i]
        # X_train = np.concatenate([X_train, x_train_i], axis = 0)
        # Y_train = np.concatenate([Y_train, y_train_i], axis = 0).astype('float32')
        X_train.append(x_train_i)
        Y_train.append(y_train_i)
        print(Y_train.shape)

select_classes_0_to_4(x_train, y_train, x_test, y_test)