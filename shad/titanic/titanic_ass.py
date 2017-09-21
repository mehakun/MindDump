import pandas
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# sexes = data['Sex'].get_values()

# unique, counts = np.unique(sexes, return_counts=True)
# result = dict(zip(unique, counts))

# male_count, female_count = result['male'], result['female']
# print(data['Sex'].value_counts(), end='')

# print('{:.2f}'.format(data['Survived'].sum() / data['Survived'].count() * 100), end='')

# print('{:.2f} {}'.format(data['Age'].mean(), data['Age'].median()), end='')

# print('{:.2f}'.format(data.corr()['SibSp']['Parch']), end='')

# females = data[data['Sex'] == 'female']['Name']
# res = females.str.replace('\w*,\s+\w*\.\s*', '').str.split(expand=True)[0].value_counts().index.get_level_values(0)
# print(res.str.replace('\s*\w*\s*]*\(', '').value_counts().index.get_level_values(0)[0], end='')
# whoops that doesnt work now))
