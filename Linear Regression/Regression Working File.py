import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

# data trim
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# 4 different arrays (uses 10% of the data that we will test it on, data different each times)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
# Find the best model in a certain range
best = 0
for _ in range(100):

    # 4 different arrays (uses 10% of the data that we will test it on, data different each times)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Model fit
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Score
    acc = linear.score(x_test, y_test)
    print(acc)

    # Saving our best model
    if acc > best:
        best = acc
        print("best:", best)
        xTest = x_test
        yTest = y_test
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
        print("saved!")
'''

# Loading saved model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# print("best:", linear.score(xTest, yTest))
print("best:", linear.score(x_test, y_test))

# Normal print
# print("Co:", linear.coef_)
# print("Intercept:", linear.intercept_)

# Pretty print
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

print("Prediction:")
for i in range(len(predictions)):
    print(round(predictions[i]), x_test[i], y_test[i])

# Plots of a correlation
p = "failures"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
