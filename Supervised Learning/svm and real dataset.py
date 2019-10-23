import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
bankdata = pd.read_csv("/media/apubra/e3371ede-8799-41bd-8f5f-9ef53c1ee7fa/Git Repository/machine-learning/datasets/bill_authentication.csv")
print(bankdata.shape)
print(bankdata.head())
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
# df = pd.DataFrame([[3.62160,8.6661,-2.8073,-0.44699]])
# y_pred = svclassifier.predict(df)
y_pred = svclassifier.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# print(y_test)
# print(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)