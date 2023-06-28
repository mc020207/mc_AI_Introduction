import numpy as np
from PIL import Image
import random
import os
def init_project():
    exists=os.path.exists
    if (not exists("X_train.npy")) or (not exists("y_train.npy")) or (not exists("X_validate.npy")) or (not exists("y_validate.npy")) or (not exists("X_test.npy")) or (not exists("y_test.npy")):
        X=np.random.rand(10000,1)*2*np.pi-np.pi
        Y=np.sin(X)
        delta_y=np.random.rand(10000,1)*0.02-0.01
        Y+=delta_y
        np.save("X_train.npy",X[:8000])
        np.save("y_train.npy",Y[:8000])
        np.save("X_validate.npy",X[8000:9000])
        np.save("y_validate.npy",Y[8000:9000])
        np.save("X_test.npy",X[9000:])
        np.save("y_test.npy",Y[9000:])
    return np.load("X_train.npy"),np.load("y_train.npy"),np.load("X_validate.npy"),np.load("y_validate.npy"),np.load("X_test.npy"),np.load("y_test.npy")