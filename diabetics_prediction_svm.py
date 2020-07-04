import sklearn
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics


data = pd.read_csv("datasets_diabetes.csv")

data = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction","Age","Outcome"]]


result = "Outcome"

x = np.array(data.drop([result], 1))
y = np.array(data[result])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


clf = svm.SVC(kernel="poly")
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
