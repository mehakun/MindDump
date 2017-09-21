import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

NEEDED_ARGS_LIST = ['Pclass', 'Sex', 'Age', 'Fare']

data = pandas.read_csv('titanic.csv', index_col='PassengerId').dropna()
survived_arr = data['Survived']
parsed_data = data.loc[:, data.columns.isin(NEEDED_ARGS_LIST)].replace(
    {NEEDED_ARGS_LIST[1] : {'male' : -1, 'female' : 1}})
parsed_data['Age'] = parsed_data['Age'].astype(np.dtype(np.int64))

# print(survived_arr)
# print(parsed_data)
dec_tree = DecisionTreeClassifier(random_state=241)
dec_tree = dec_tree.fit(parsed_data, survived_arr)
print(parsed_data.columns)
print(dec_tree.feature_importances_)
importances = sorted(zip(parsed_data.columns, dec_tree.feature_importances_),
                     key = lambda t: t[1])

print(importances)
print('{0} {1}'.format(importances[-1][0], importances[-2][0]), end='')
