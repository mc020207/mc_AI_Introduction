import numpy as np
from PIL import Image
import random
import os
from torchvision.transforms import Resize
from MyDataset import MyDataset


def init_project():
    transforms = Resize(224)
    exists = os.path.exists
    # if (not exists("X_train.npy")) or (not exists("y_train.npy")) or (not exists("X_validate.npy")) or (
    #         not exists("y_validate.npy")) or (not exists("X_test.npy")) or (not exists("y_test.npy")):
    X = []
    for i in range(12):
        X_temp = []
        for j in range(240):
            path = "H:/test_data/" + str(i + 1) + "/" + str(j + 1) + ".bmp"
            img = np.asfarray(Image.open(path))
            img = img.reshape(1, -1)
            if j == 0:
                X_temp = img
            else:
                X_temp = np.vstack((X_temp, img))
        if i == 0:
            X = X_temp
        else:
            X = np.vstack((X, X_temp))
    X = 1 - X
    # np.save("X.npy", X)
    Y = np.array([int(x // 240) for x in range(240 * 12)], dtype=np.int)
    print(Y.shape)
    c = list(zip(X, Y))
    random.shuffle(c)
    X, Y = zip(*c)
    Y = np.array(Y, dtype=np.int)
    X = np.array(X)
        # np.save("X_train.npy", X[:6000])
        # np.save("y_train.npy", Y[:6000])
        # np.save("X_validate.npy", X[6000:7200])
        # np.save("y_validate.npy", Y[6000:7200])
        # np.save("X_test.npy", X[7200:])
        # np.save("y_test.npy", Y[7200:])
    # X_train = np.load("X_train.npy")
    # y_train = np.load("y_train.npy")
    # X_validate = np.load("X_validate.npy")
    # y_validate = np.load("y_validate.npy")
    # X_test = np.load("X_test.npy")
    # y_test = np.load("y_test.npy")
    trainSet = [X, Y]
    validateSet = [X, Y]
    testSet = [X, Y]
    return MyDataset(trainSet, transforms), MyDataset(validateSet, transforms), MyDataset(testSet, transforms)


# def init_project():
#     transforms = Resize(32)
#     X = []
#     for i in range(12):
#         X_temp = []
#         for j in range(240):
#             path = "H:/test_data/" + str(i + 1) + "/" + str(j + 1) + ".bmp"
#             img = np.asfarray(Image.open(path))
#             img = img.reshape(1, -1)
#             if j == 0:
#                 X_temp = img
#             else:
#                 X_temp = np.vstack((X_temp, img))
#         if i == 0:
#             X = X_temp
#         else:
#             X = np.vstack((X, X_temp))
#     X = 1 - X
#     Y = np.array([int(x // 240) for x in range(240 * 12)], dtype=np.int)
#     print(Y.shape)
#     c = list(zip(X, Y))
#     random.shuffle(c)
#     X, Y = zip(*c)
#     Y = np.array(Y, dtype=np.int)
#     X = np.array(X)
#     X_train = X
#     y_train = Y
#     X_validate = X
#     y_validate = Y
#     X_test = X
#     y_test = Y
#     trainSet = [X_train, y_train]
#     validateSet = [X_validate, y_validate]
#     testSet = [X_test, y_test]
#     return MyDataset(trainSet, transforms), MyDataset(validateSet, transforms), MyDataset(testSet,transforms)