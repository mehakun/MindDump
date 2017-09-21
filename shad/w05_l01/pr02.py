import numpy as np
import pandas
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

def sigmoid(y_pred):
    return 1. / (1. + np.exp(-y_pred))

data = pandas.read_csv('gbm-data.csv')
data_vec = data['Activity']
data_matrix = data.loc[:, 'D1':]
train_matrix, test_matrix, train_vec, test_vec = train_test_split(data_matrix, data_vec, test_size=0.8, random_state=241)

# LEARNING_RATES = (1, 0.5, 0.3, 0.2, 0.1)
# min_metric, iteration = 10, -1
# -------------------------------------------------------------------------------------------------------------------------
# Plots graphics of log_loss of training and testing sets and also prints minimum measure. Test set is overfitted
# list_of_mins = []

# for learning_rate in LEARNING_RATES:
#     print('current rate is', learning_rate, end='\n')
#     gbc = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
#     gbc.fit(train_matrix, train_vec)
#     train_loss, test_loss = [], []
    
#     for i, pred in enumerate(gbc.staged_decision_function(train_matrix)):
#         train_loss.append(log_loss(train_vec, sigmoid(pred)))
#     for i, pred in enumerate(gbc.staged_decision_function(test_matrix)):
#         test_loss.append(log_loss(test_vec, sigmoid(pred)))
#     train_min, test_min = np.amin(train_loss), np.amin(test_loss)
    
#     if train_min < test_min and train_min < min_metric:
#         min_metric = train_min
#         iteration = train_loss.index(min_metric)
#     elif test_min < min_metric:
#         min_metric = test_min
#         iteration = test_loss.index(min_metric)
#     print('iter = {} val = {:.2}'.format(iteration, test_min))
#     list_of_mins.append(test_min)
    
#     plt.figure()
#     plt.plot(test_loss, 'r', linewidth=2)
#     plt.plot(train_loss, 'g', linewidth=2)
#     plt.legend(['test', 'train'])
#     plt.show()
# print('FINAL:\niter = {} val = {:.2}'.format(iteration, test_min))
# -------------------------------------------------------------------------------------------------------------------------
gbc = GradientBoostingClassifier(n_estimators=250, random_state=241, learning_rate=0.2)
gbc.fit(train_matrix, train_vec)
test_loss = []
iter_list = []
for i, pred in enumerate(gbc.staged_decision_function(test_matrix)):
    iter_list.append(i)
    test_loss.append(log_loss(test_vec, sigmoid(pred)))

test_min = np.amin(test_loss)
# prints iteration with lowest loss (loss = 0.53, iteration = 36)
# print('{:.2} {}'.format(test_min, iter_list[test_loss.index(test_min)]), end='')
# -------------------------------------------------------------------------------------------------------------------------
# find log loss of rfc's prediction using iterations found in prev task as amount of estimators
rfc = RandomForestClassifier(n_estimators=iter_list[test_loss.index(test_min)], random_state=241)
rfc.fit(train_matrix, train_vec)
print('{:.2}'.format(log_loss(test_vec, rfc.predict_proba(test_matrix))), end='') # log loss is 0.54
