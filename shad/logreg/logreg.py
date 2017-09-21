import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
data = pandas.read_csv('data-logistic.csv', header=None)
y = data.iloc[:, 0]
X1 = data.iloc[:, 1]
X2 = data.iloc[:, 2]
l = len(y)

def eucl(X, Y):
    return np.sqrt(np.sum([(X[i] - Y[i]) ** 2 for i in range(len(X))]))

def grad_descent(W_prev, k=0.1, C=10):
    print('In desc:')
    print(W_prev)
    w1 = W_prev[0] + k / l * np.sum(
        [y[i] * X1[i] * (1. - 1. / (1. + np.exp(-y[i] * (W_prev[0] * X1[i] + W_prev[1] * X2[i])))) for i in range(l)]) - k * C * W_prev[0]
    w2 = W_prev[1] + k / l * np.sum(
        [y[i] * X2[i] * (1. - 1. / (1. + np.exp(-y[i] * (W_prev[0] * X1[i] + W_prev[1] * X2[i])))) for i in range(l)]) - k * C * W_prev[1]
    print([w1, w2])
    return np.array([w1, w2])

def sigmoid(W):
    return 1. / (1. + np.exp(-W[0] * X1 - W[1] * X2))

def log_reg(W_prev=np.array([0, 0]), steps=10000, eps=1e-5, C=10):
    for step in range(steps):
        w = grad_descent(W_prev, C=C)
        diff = eucl(w, W_prev)
        print(diff)
        if (diff <= eps):
            print('finished')
            return w
        W_prev = np.copy(w)

        print(step)
print('{:.3} {:.3}'.format(roc_auc_score(y, sigmoid(log_reg(C=10))), roc_auc_score(y, sigmoid(log_reg(C=0)))))
