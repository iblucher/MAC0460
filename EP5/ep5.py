import numpy as np
from keras.datasets import mnist

from dataset_assembly import create_dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dataset creation function
X_train, Y_train, X_test, Y_test = create_dataset(x_train, y_train, x_test, y_test)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Shuffle data