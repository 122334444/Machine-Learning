import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matlablib.colors import listedColormap
from KNN import KNN

camp=listedColormap(['#FF0000','#00FF00','#0000FF'])

iris=datasets.load_iris()
X,y = iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


plt.figure()
plt.scatter(X_train[:,2],X_train[:,3],c=y_train,cmap=camp,edgecolors='k',s=20)
plt.show()

clf = KNN(k=5)

clf.fit(X_train,y_train)
pridictions = clf.predict(X_test)

print("Predictions:",pridictions)