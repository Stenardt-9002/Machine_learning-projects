from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


cancer_data = datasets.load_breast_cancer()
# print(cancer_data)

X_train ,X_test ,y_train,y_test = train_test_split(cancer_data.data,cancer_data.target,test_size=0.4,random_state=209)
# print(X_test)
# print(X_train)
# print(y_test)
# print(y_train)

cls = svm.SVC(kernel="linear")
cls.fit(X_train,y_train)

pred = cls.predict(X_test)

print("accuracy:",metrics.accuracy_score(y_test,y_pred=pred))
print("precision",metrics.precision_score(y_test,y_pred=pred))
print("recall",metrics.recall_score(y_test,y_pred=pred))
print(metrics.classification_report(y_test,y_pred=pred))


# pri       nt(cancer_data)