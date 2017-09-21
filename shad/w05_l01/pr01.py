import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
data_matrix = data.loc[:, 'Sex':'ShellWeight'].as_matrix()
data_vec = data['Rings'].as_matrix()

kfold = KFold(n_splits=5, random_state=1, shuffle=True)

for i in range(1, 51):
    rfr = RandomForestRegressor(n_estimators=i, random_state=1)
    scores = 0
    
    for train_indices, test_indices in kfold.split(data_matrix, data_vec):
        rfr.fit(data_matrix[train_indices], data_vec[train_indices])
        pred = rfr.predict(data_matrix[test_indices])
        scores += r2_score(data_vec[test_indices], pred)

    if scores / 5 > 0.52:
        print(i, end='')
        exit();
