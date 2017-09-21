import numpy as np
import pandas
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

#data = pandas.read_csv('classification.csv')
#v_true = data['true']
#v_pred = data['pred']

#print(accuracy_score(v_true, v_pred), precision_score(v_true, v_pred), recall_score(v_true, v_pred), f1_score(v_true, v_pred))

#conf_matrix = confusion_matrix(v_true, v_pred)
#print(conf_matrix[1][1], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[0][0])

data = pandas.read_csv('scores.csv')
classes = data.iloc[:, 1:]

#res = [(roc_auc_score(data['true'], classes[column]), column) for column in classes]
#print('{}'.format(max(res)[1]), end='')
max_column = ''
max_precision = -1.

for column in classes:
    precision, recall, thresholds = precision_recall_curve(data['true'], classes[column])
    i = 0
    
    while recall[i] >= 0.7:
        if precision[i] > max_precision:
            max_column = column
            max_precision = precision[i]
        i += 1
            
print(max_column, end='')
#    print(dict_list)
#    print(tmp_list)



