#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Thursday Jan 30th 2019
author: Murari Goswami
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


# loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

# build the model
clf = RandomForestClassifier(n_estimators=10)

# train the classifier
clf.fit(X_train, y_train)

# predictions
predicted = clf.predict(X_test)

# check accuracy
print(accuracy_score(predicted, y_test))

# create pickle file
with open('rf.pkl', 'wb') as model_pk:
    pickle.dump(clf, model_pk)
