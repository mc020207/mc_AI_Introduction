import numpy as np
from PIL import Image
import random
import os
def init_project():
    prenum=240
    exists=os.path.exists
    if (not exists("X_train.npy")) or (not exists("y_train.npy")) or (not exists("X_validate.npy")) or (not exists("y_validate.npy")) or (not exists("X_test.npy")) or (not exists("y_test.npy")):
        X=[]
        for i in range(12):
            X_temp=[]
            for j in range(prenum):
                path="H:/test_data/"+str(i+1)+"/"+str(j+1)+".bmp"
                img = np.asfarray(Image.open(path))
                img=img.reshape(1,-1)
                if j==0:
                    X_temp=img
                else:
                    X_temp=np.vstack((X_temp,img))
            if i==0:
                X=X_temp
            else:
                X=np.vstack((X,X_temp))
        X=1-X
        np.save("X.npy",X)
        Y=np.array([int(x//prenum) for x in range(prenum*12)],dtype=np.int)
        print(Y.shape)
        c = list(zip(X,Y))
        random.shuffle(c)
        X,Y = zip(*c)
        Y=np.array(Y,dtype=np.int)
        X=np.array(X)
        np.save("X_train.npy",X[:6000])
        np.save("y_train.npy",Y[:6000])
        np.save("X_validate.npy",X[6000:7200])
        np.save("y_validate.npy",Y[6000:7200])
        np.save("X_test.npy",X[0:])
        np.save("y_test.npy",Y[0:])
    return np.load("X_train.npy"),np.load("y_train.npy"),np.load("X_validate.npy"),np.load("y_validate.npy"),np.load("X_test.npy"),np.load("y_test.npy")