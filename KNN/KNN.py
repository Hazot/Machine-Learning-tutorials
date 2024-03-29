# Learning about KNN (K needs to be an odd number)
# Euclidian distance

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

# importing the data
data = pd.read_csv("car.data")
print(data.head())
print("=======================")
print(data.tail())
print("=======================")


# Encode the data into integers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# print(safety)

# Combine our data into the same label with zip (combines the data into a tuple)
x = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["acc", "good", "unacc", "vgood"]

for i in range(len(predicted)):
    # print("tabarnak: ", predicted[i])
    # print("==========================")
    print("Predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9, True)
    # print("N: ", n)
