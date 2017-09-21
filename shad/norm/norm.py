import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

learn_data = pandas.read_csv('perceptron-train.csv', header=None)
learn_vec, learn_matrix = learn_data.loc[:, 0], learn_data.loc[:, 1:]
# print(learn_vec, learn_matrix)
test_data = pandas.read_csv('perceptron-test.csv', header=None)
test_vec, test_matrix = test_data.loc[:, 0], test_data.loc[:, 1:]
# print(test_vec, test_matrix)

perceptron = Perceptron(random_state=241)
perceptron.fit(learn_matrix, learn_vec)

before_norm = accuracy_score(test_vec, perceptron.predict(test_matrix))
# print('before norm = ', before_norm)
scaler = StandardScaler()
learn_matrix = scaler.fit_transform(learn_matrix)
test_matrix = scaler.transform(test_matrix)

perceptron.fit(learn_matrix, learn_vec)

after_norm = accuracy_score(test_vec, perceptron.predict(test_matrix))
# print('after norm = ', after_norm)
print('{:.3f}'.format(after_norm - before_norm), end='')
