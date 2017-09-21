import pandas
import numpy as np
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None)
vec_learn = data[0]
matrix_learn = data.loc[:, 1:]

svc = SVC(C=100000, random_state=241)
svc.fit(matrix_learn, vec_learn)

print(','.join([str(x + 1) for x in svc.support_]), end='')
                 
