import numpy as np

test_matrix = np.random.normal(loc=1, scale=10, size=(1000, 50))
print(test_matrix)
print()

mean = np.mean(test_matrix, axis=0)
std = np.std(test_matrix, axis=0)
norm_matrix = (test_matrix - mean) / std
print(norm_matrix)
print()

Z = np.array([[4, 5, 0], 
              [1, 9, 3],              
              [5, 1, 1],
              [3, 3, 3], 
              [9, 9, 9], 
              [4, 7, 1]])
sum_matrix = np.sum(Z, axis=1)
print(np.nonzero(sum_matrix > 10))
print()

A = np.eye(N=3)
B = np.eye(N=3)
join_matrix = np.vstack((A, B))
print(join_matrix)
print()
