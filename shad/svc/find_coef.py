import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
import math

newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism', 'sci.space']
)
vectorizer = TfidfVectorizer(min_df=1)
val_matrix = vectorizer.fit_transform(newsgroups.data)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(val_matrix, newsgroups.target)

max_mean, c_val = 0, 0
#  cv_results_
for a in gs.grid_scores_:
    if a.mean_validation_score > max_mean:
        max_mean = a.mean_validation_score
        c_val = a.parameters['C']

clf = SVC(C=c_val, kernel='linear', random_state=241)
clf.fit(val_matrix, newsgroups.target)

coef = clf.coef_

feature_mapping = dict(zip(vectorizer.get_feature_names(), coef.transpose().toarray()))

result = sorted(feature_mapping.items(), key=lambda x: np.fabs(x[1]), reverse=True)[:10]
result.sort()
print(','.join(x[0] for x in result), end='')



