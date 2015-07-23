#!/usr/bin/python

import csv
import numpy as np
from sklearn import svm

# 0: PassengerId, 1: Survived, 2: Pclass, 3: Name, 4: Sex (male,female), 5: Age, 6: SibSp, 7: ParCh, 8: Ticket, 9: Fare, 10: Cabin, 11: Embarked (S,C,Q)
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
def loadData(inputFile, means = [], stds = []):
  X = [] # features
  y = [] # labels
  missingCount = 0
  reader = csv.reader(open(inputFile, 'rb'))
  headers = reader.next()
  
  m = {"S": 1, "C": 2, "Q": 3, "": 4}
  for row in reader:
    y.append(int(row[1]))
    pclass = int(row[2])
    sex = 1 if row[4] == "male" else 0
    age = float(row[5]) if row[5] != "" else 25
    sibsp = int(row[6])
    parch = int(row[7])
    fare = float(row[9])
    embarkeds = 1 if row[11] == "S" else 0
    embarkedc = 1 if row[11] == "C" else 0
    embarkedq = 1 if row[11] == "Q" else 0
    
    X.append([pclass, sex, age, sibsp, parch, fare, embarkeds, embarkedc, embarkedq])
  
  # scale each feature to standard normal
  X = np.array(X)
  calcNormalization = False
  if len(means) == 0:
    calcNormalization = True
  numFeatures = X.shape[1]
  for i, col in enumerate(range(numFeatures)):
    a = X[:,col]
    if calcNormalization:
      means.append(np.mean(a))
      stds.append(np.std(a))
    X[:,col] = (a - means[i]) / stds[i]
  
  if calcNormalization:
    return X, y, means, stds
  else:
    return X, y

trainFile = '../data/train_validation.csv'
X, y, means, stds = loadData(trainFile)

print "#Passengers", len(y)
print "Survived", sum(y)
print "Deceased", len(y) - sum(y)

testFile = '../data/test_validation.csv'
XTest, yTest = loadData(testFile, means, stds)

linear_clf = svm.SVC(kernel='linear')
linear_clf.fit(X, y)

yTestPredicted = linear_clf.predict(XTest)
yTrainPredicted = linear_clf.predict(X)

print ""
print "Linear SVM"
print "Correctly predicted Train:", sum(y == yTrainPredicted)
print "Incorrectly predicted Train:", sum(y != yTrainPredicted)
print "Correctly predicted Test:", sum(yTest == yTestPredicted)
print "Incorrectly predicted Test:", sum(yTest != yTestPredicted)
print "Weights:", linear_clf.coef_

rbf_clf = svm.SVC(kernel='rbf')
rbf_clf.fit(X, y)

yTestPredicted = rbf_clf.predict(XTest)
yTrainPredicted = rbf_clf.predict(X)

print ""
print "RBF SVM"
print "Correctly predicted Train:", sum(y == yTrainPredicted)
print "Incorrectly predicted Train:", sum(y != yTrainPredicted)
print "Correctly predicted Test:", sum(yTest == yTestPredicted)
print "Incorrectly predicted Test:", sum(yTest != yTestPredicted)

sigmoid_clf = svm.SVC(kernel='sigmoid')
sigmoid_clf.fit(X, y)

yTestPredicted = sigmoid_clf.predict(XTest)
yTrainPredicted = sigmoid_clf.predict(X)

print ""
print "Sigmoid SVM"
print "Correctly predicted Train:", sum(y == yTrainPredicted)
print "Incorrectly predicted Train:", sum(y != yTrainPredicted)
print "Correctly predicted Test:", sum(yTest == yTestPredicted)
print "Incorrectly predicted Test:", sum(yTest != yTestPredicted)

poly2_clf = svm.SVC(kernel='poly', degree = 2)
poly2_clf.fit(X, y)

yTestPredicted = poly2_clf.predict(XTest)
yTrainPredicted = poly2_clf.predict(X)

print ""
print "Poly(2) SVM"
print "Correctly predicted Train:", sum(y == yTrainPredicted)
print "Incorrectly predicted Train:", sum(y != yTrainPredicted)
print "Correctly predicted Test:", sum(yTest == yTestPredicted)
print "Incorrectly predicted Test:", sum(yTest != yTestPredicted)

poly3_clf = svm.SVC(kernel='poly')
poly3_clf.fit(X, y)

yTestPredicted = poly3_clf.predict(XTest)
yTrainPredicted = poly3_clf.predict(X)

print ""
print "Poly(3) SVM"
print "Correctly predicted Train:", sum(y == yTrainPredicted)
print "Incorrectly predicted Train:", sum(y != yTrainPredicted)
print "Correctly predicted Test:", sum(yTest == yTestPredicted)
print "Incorrectly predicted Test:", sum(yTest != yTestPredicted)

poly5_clf = svm.SVC(kernel='poly', degree = 5)
poly5_clf.fit(X, y)

yTestPredicted = poly5_clf.predict(XTest)
yTrainPredicted = poly5_clf.predict(X)

print ""
print "Poly(5) SVM"
print "Correctly predicted Train:", sum(y == yTrainPredicted)
print "Incorrectly predicted Train:", sum(y != yTrainPredicted)
print "Correctly predicted Test:", sum(yTest == yTestPredicted)
print "Incorrectly predicted Test:", sum(yTest != yTestPredicted)
