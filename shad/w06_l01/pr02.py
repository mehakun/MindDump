import numpy as np
from sklearn.cluster import KMeans
from skimage import img_as_float
from skimage.io import imread
from sklearn.metrics import mean_squared_error
import pylab
# %matplotlib inline

def PSNR(y_true, y_pred):
    return 10. * np.log10(1. / np.mean((y_true - y_pred) ** 2))

def fill_with_mean(matrix, km):
    X_mean = np.copy(matrix)
    for i, value in enumerate(km.cluster_centers_):        
        X_mean[km.labels_ == i] = value
    return X_mean

def fill_with_med(X, clf):
    X_median = np.copy(X)
    for i in range(clf.n_clusters):
        X_median[clf.labels_ == i] = np.median(X[clf.labels_ == i], axis=0)
    return X_median

image = img_as_float(imread('parrots.jpg'))
x, y, z = image.shape
image_matrix = np.reshape(image, (x * y, z))

#km = KMeans(init='k-means++', random_state=241).fit(matrix_to_mod)

#print(PSNR(image_matrix, matrix_to_mod))
for clusters_amount in range(2, 21):
    clf = KMeans(n_clusters=clusters_amount, init='k-means++', random_state=241).fit(image_matrix)
    matrix_mean = fill_with_mean(image_matrix, clf)
    matrix_median = fill_with_med(image_matrix, clf)

    mean_view = np.reshape(matrix_mean, (x, y, z))
    median_view = np.reshape(matrix_median, (x, y, z))
    print('Mean View')
    pylab.imshow(mean_view)
    pylab.show()
    print('Median View')
    pylab.imshow(median_view)
    pylab.show()
    
    psnr_mean = PSNR(image_matrix, matrix_mean)
    psnr_med = PSNR(image_matrix, matrix_median)
    
    print('mean {} {}'.format(clusters_amount, psnr_mean), end='\n')

    print('median {} {}'.format(clusters_amount, psnr_med), end='\n')
        #        exit()
