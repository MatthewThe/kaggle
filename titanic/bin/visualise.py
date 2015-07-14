#!/usr/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

idx = 0
# 0: PassengerId, 1: Survived, 2: Pclass, 3: Name, 4: Sex (male,female), 5: Age, 6: SibSp, 7: ParCh, 8: Ticket, 9: Fare, 10: Cabin, 11: Embarked (S,C,Q)
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
def plotHist(col, edges, title):
  global idx
  x = []
  y = []
  missingCount = 0
  reader = csv.reader(open('../data/train.csv', 'rb'))
  headers = reader.next()
  
  for row in reader:
    y.append(int(row[1]))
    if len(row[col]) > 0:
      if col == 4:
        row[col] = 1 if row[col] == "male" else 0
      elif col == 11:
        m = {"S": 1, "C": 2, "Q": 3}
        row[col] = m[row[col]]
      x.append(float(row[col]))
    else:
      x.append(-8.0)
      missingCount += 1
  
  
  if idx == 0:
    print "Total"
    print "#Passengers", len(y)
    print "Survived", sum(y)
    print "Deceased", len(y) - sum(y)
  
  print ""
  print title
  print "#Missing value", missingCount
  
  idx += 1
  plt.subplot(3,3,idx)
  plt.title(title)
  plt.hist([a for a,b in zip(x,y) if b == 1], bins = edges, alpha = 0.5, label = 'survived')
  plt.hist([a for a,b in zip(x,y) if b == 0], bins = edges, alpha = 0.5, label = 'deceased')
  plt.legend()

# Pclass
plotHist(2, np.arange(0.5,4.5,1), "Pclass")

# Gender
plotHist(4, np.arange(-0.5,2.5,1), "Gender")

# Age
plotHist(5, range(-10,100,5), "Age")

# SibSp
plotHist(6, np.arange(-0.5,9.5,1), "Siblings/Spouse")

# ParCh
plotHist(7, np.arange(-0.5,9.5,1), "Parents/Chilren")

# Fare
plotHist(9, range(0,200,10), "Fare")

# Fare
plotHist(11, np.arange(0.5,4.5,1), "Embarked")
  
plt.show()
