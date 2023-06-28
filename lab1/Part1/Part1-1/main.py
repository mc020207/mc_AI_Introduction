import numpy as np
from net import MLPClassifier
from matplotlib import pyplot as plt
from init import init_project
X_train,y_train,X_validate,y_validate,X_test,y_test=init_project()
print('init OK!')
obj=MLPClassifier(1,1)
save_path="mpl_best_model4.pdparams"
obj.fit(X_train,y_train,X_validate,y_validate,lr=0.02,num_epochs=5000,log_epochs=100,save_path=save_path)

