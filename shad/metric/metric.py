import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

def find_best(matrix, vec, cv_generator):
    index_max, max_val = 0, 0

    for i in range(1, 51):
        result = np.mean(cross_val_score(estimator=KNeighborsClassifier(n_neighbors=i),
                                         X=matrix, y=vec, cv=cv_generator))
        print(i, result)
        if result > max_val:
            index_max, max_val = i, result

    return index_max, max_val


data = pandas.read_csv('wine.data', header=None)
class_vec = data[0]
attr_matrix = data.loc[:, 1:]

f1, f2, f3, f4 = open('f1', 'w'), open('f2', 'w'), open('f3', 'w+'), open('f4', 'w')

KFold = KFold(n_splits=5, random_state=42, shuffle=True)

index_max, max_val = find_best(attr_matrix, class_vec, KFold)

f1.write(str(index_max))
f2.write('{:.2f}'.format(max_val))

attr_matrix = scale(attr_matrix)
index_max, max_val = find_best(attr_matrix, class_vec, KFold)

f3.write(str(index_max))
f4.write('{:.2f}'.format(max_val))
