import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

dataset = loadmat('olivettifaces.mat')
faces = dataset['faces']

# PCA
pca = PCA(n_components=3)
data = pca.fit_transform(faces)
print(data)


# Manual PCA
