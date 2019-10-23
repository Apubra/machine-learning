import pandas as pd
import numpy as np
import random as rnd
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../datasets/titanic/train.csv')
test_df = pd.read_csv('../datasets/titanic/test.csv')
train_test_data = [train_df, test_df]
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


X_train = train_df.drop(['Name', 'PassengerId', 'Survived','Sex'], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(['Name', 'PassengerId','Sex'], axis=1).copy()

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree