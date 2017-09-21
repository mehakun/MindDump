import numpy as np
import pandas
from sklearn.decomposition import PCA

data = pandas.read_csv('close_prices.csv')
data_wo_date = data.drop('date', 1)
# do we really need date column? dropped it
pca = PCA(n_components=10)
# 4 components are enough to cover > 90% of variance
pca.fit(data_wo_date)
# print(np.sum(pca.explained_variance_ratio_))
# -------------------------------------------------------------------
# get first component's values (first column)
first_component = pca.transform(data_wo_date)[:, 0]
# load dj index without date column and convert it to matrix
djia_index = pandas.read_csv('djia_index.csv')['^DJI'].as_matrix()
# find Pearson correlation between dj index and first_component
corrcoef = np.corrcoef(first_component, djia_index)
#print('{:.2}'.format(corrcoef[0][1]), end='')
# -------------------------------------------------------------------
# find company with biggest weight
print(data_wo_date.columns.values[np.argmax(pca.components_[0])], end='') # it's V
