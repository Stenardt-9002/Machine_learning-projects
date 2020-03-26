import random
import matplotlib.pyplot as plt
from sklearn import datasets


from sklearn import svm

datadig = datasets.load_digits()

# print(datadig.data )
# print(datadig.images[0])#print 0



#......................................
clf = svm.SVC(gamma = 0.001,C = 100)

X,y = datadig.data[:-1],datadig.target[:-1]

clf.fit(X,y)

# print(len(datadig.data))

# print(clf.predict(datadig.data[-1].reshape(1,-1)))
# plt.imshow(datadig.images[-1],interpolation="nearest")

i = random.randint(1,1700)
print(clf.predict(datadig.data[i].reshape(1,-1)))
plt.imshow(datadig.images[i],interpolation="nearest")
plt.show()
