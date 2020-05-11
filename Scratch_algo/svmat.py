import numpy as np 
# import tensorflow as tf
# from tensorflow.keras.wrappers.scikit_learn import svm ,cross_validation
from sklearn import preprocessing,neighbors,svm
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validation
import pandas as pd



df1 = pd.read_csv('data.csv')
df1.replace('?',-99999,inplace = True)
df1.drop(['id'],1,inplace = True)


y = np.array(df1['class'])
X = np.array(df1.drop(['class'],1))

X_train ,X_test ,y_train,y_test = train_test_split(X,y,test_size = 0.2)


clf =  svm.SVC()
clf.fit(X_train,y_train)

accu_racy = clf.score(X_test,y_test)
print(accu_racy)

example_measure = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measure = example_measure.reshape(len(example_measure),-1)

prediction = clf.predict(example_measure)
print(prediction)







