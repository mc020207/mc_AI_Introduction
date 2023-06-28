import numpy as np
from net import MLPClassifier
from matplotlib import pyplot as plt
from init import init_project
X_train,y_train,X_validate,y_validate,X_test,y_test=init_project()
print('init OK!')
idx=29
obj=MLPClassifier(28*28,12)
save_path="mpl_best_model"+str(idx)+".pdparams"
obj.fit(X_train,y_train,X_validate,y_validate,lr=0.05,num_epochs=100,log_epochs=10,save_path=save_path)
obj.fit(X_train,y_train,X_validate,y_validate,lr=0.02,num_epochs=30,log_epochs=10,save_path=save_path)
obj.fit(X_train,y_train,X_validate,y_validate,lr=0.01,num_epochs=30,log_epochs=10,save_path=save_path)
print(obj.predict(X_test,y_test,save_path))