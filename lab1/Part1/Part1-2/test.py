import numpy as np
from net import MLPClassifier
from matplotlib import pyplot as plt
from init import init_project
X_train,y_train,X_validate,y_validate,X_test,y_test=init_project()
# for idx in range(1,29):
#     obj=MLPClassifier(28*28,12)
#     save_path="mpl_best_model"+str(idx)+".pdparams"
#     print(save_path,":",obj.predict(X_test,y_test,save_path))
obj=MLPClassifier(28*28,12)
save_path="mpl_best_model9.pdparams"
print(obj.predict(X_test,y_test,save_path))