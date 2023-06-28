import numpy as np
from net import MLPClassifier
from matplotlib import pyplot as plt
from init import init_project
X_train,y_train,X_validate,y_validate,X_test,y_test=init_project()
obj=MLPClassifier(1,1)
save_path="mpl_best_model2.pdparams"
y_test=obj.predict(X_test,y_test,save_path)
X_true = np.linspace(-np.pi,np.pi,num=1000)
y_true = np.sin(X_true)
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.scatter(X_test, y_test, facecolor="none", edgecolor='#e4007f', s=5, label="train data")
plt.plot(X_true, y_true, c='#000000', label=r"$\sin(2\pi x)$")
plt.legend(fontsize='x-large')
plt.savefig('ml-vis2.pdf')
plt.show()