import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[3, -1]]))

# accuracy
from sklearn.metrics import accuracy_score
test_points = [[1, -1], [2, 2], [3, 3], [4, 3]]
test_labels = [1, 2, 2,2]
predicts = clf.predict(test_points)
print(predicts)
accuracy = accuracy_score(test_labels,predicts)
print(accuracy)