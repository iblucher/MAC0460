import numpy as np
import itertools

from keras.datasets import mnist
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from dataset_assembly import create_dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dataset creation function
X_train, Y_train, X_test, Y_test = create_dataset(x_train, y_train, x_test, y_test)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Shuffle data
X_train, Y_train = shuffle(X_train, Y_train)

# 5-fold cross-validation
k = 5
len_dataset = X_train.shape[0]
fold_size = len_dataset // k

x_cv_folds = [X_train[i:i+fold_size] for i in range(0, len_dataset, fold_size)]
y_cv_folds = [Y_train[i:i+fold_size] for i in range(0, len_dataset, fold_size)]

folds_accuracy_svm = []
folds_accuracy_mlp = []
for fold in range(k):
    X_test_cv = x_cv_folds[fold]
    Y_test_cv = y_cv_folds[fold]

    x_train_cv = [x for i,x in enumerate(x_cv_folds) if i != fold]
    X_train_cv = list(itertools.chain.from_iterable(x_train_cv))
    y_train_cv = [y for i,y in enumerate(y_cv_folds) if i != fold]
    Y_train_cv = list(itertools.chain.from_iterable(y_train_cv))

    # SVM
    svm = SVC(kernel='rbf', gamma=0.05, C=5.0)
    svm.fit(X_train_cv, Y_train_cv)
    Y_pred_svm = svm.predict(X_test_cv)
    accuracy_svm = accuracy_score(Y_test_cv, Y_pred_svm)
    folds_accuracy_svm.append(accuracy_svm)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,100), learning_rate_init=0.001)
    mlp.fit(X_train_cv, Y_train_cv)
    Y_pred_mlp = mlp.predict(X_test_cv)
    accuracy_mlp = accuracy_score(Y_test_cv, Y_pred_mlp)
    folds_accuracy_mlp.append(accuracy_mlp)

print(folds_accuracy_svm)
print(folds_accuracy_mlp)






    

