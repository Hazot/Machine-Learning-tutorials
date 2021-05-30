import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("target: ", cancer.target_names)

x = cancer.data
y = cancer.target

# Don,t go more than 30%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["malignant", "benign"]

clf = svm.SVC(kernel="linear", C=2)
# clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

clf2 = KNeighborsClassifier(n_neighbors=11)
clf2.fit(x_train, y_train)

y_pred2 = clf2.predict(x_test)
acc2 = metrics.accuracy_score(y_test, y_pred2)
print(acc2)