import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
import pylab as pl

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 4, 8
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

from FasterFactorAnalysis import FactorAnalyzer

fa = FactorAnalyzer(16, num_iterations=10)
t0 = time()
fa.fit(faces.T)
train_time = (time() - t0)
print 'done in %0.3f' % train_time
pl.figure(figsize=(2. * n_col, 2.26 * n_row))

for i in range(16):
    pl.subplot(n_row, n_col, i*2+1)
    comp = fa.M + 2*fa.Phi[:,i]
    pl.imshow( comp.reshape(image_shape), cmap=plt.cm.gray,
            interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
    pl.subplot(n_row, n_col, i*2+2)
    comp = fa.M - 2*fa.Phi[:,i]
    pl.imshow( comp.reshape(image_shape), cmap=plt.cm.gray,
            interpolation='nearest')
    pl.xticks(())
    pl.yticks(())


pl.show()
