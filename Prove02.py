# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:05:56 2019

@author: ausca
"""


import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
iris = datasets.load_iris()

X = iris.data
Y = iris.target

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

def euclid_distance(X, Y):
        diff_squared = (X - Y) ** 2
        distance = sum(diff_squared)
        return distance

class KNNClassifier:
    
    
    def fit(self, data_train, targets_train):
        self.data = data_train
        self.targets = targets_train
        
    def predict(self, test_train, k):
        results = []
        for x in test_train:
            distances = []
            nearest = []
            for y in self.data:
                distances.append(euclid_distance(x, y))
            closeness = np.argsort(distances)
            for temp in closeness:
                if distances[temp] < k:
                    nearest.append(self.targets[temp])
            
            most_common, frequency = Counter(nearest).most_common(1)[0]
            results.append(most_common)
        return results
    
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
classifier_two = KNNClassifier()
classifier_two.fit(X_train, Y_train)
predicted_targets = classifier_two.predict(X_test, 3)
print("KNN Library predictions:")
print(predictions)
print("My KNN Predictions:")
print(predicted_targets)
print("Actual Results:")
print(Y_test)
        