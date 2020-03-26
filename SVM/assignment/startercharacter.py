import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
dtigts = datasets.load_digits()
print(dtigts)
clf = svm.SVC(gamma=0.001,C=100)
X,y = dtigts.data[:-10],dtigts.target[:-10]
clf.fit(X,y)#train

print(clf.predict(dtigts.data[:-10]))
plt.imshow(dtigts.images[9],interpolation='nearest')
plt.show()
