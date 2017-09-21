import numpy as np
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

def find_best(matrix, vec, cv_generator):
    index_max, max_val = 0, -np.inf
    lin_space = np.linspace(1.0, 10.0, num=200)
    
    for i in lin_space:
        result = np.mean(cross_val_score(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance', p=i),
                                         X=matrix, y=vec, cv=cv_generator, scoring='neg_mean_squared_error'))
        print(i, result)
        if result > max_val:
            index_max, max_val = i, result

    return index_max


data = load_boston(return_X_y=True)
attr_matrix, class_vec = data[0], data[1]

KFold = KFold(n_splits=5, random_state=42, shuffle=True)

max_p = find_best(scale(attr_matrix), class_vec, KFold)
print(max_p, end='')
