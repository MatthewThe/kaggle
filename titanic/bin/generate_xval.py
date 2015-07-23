#!/usr/bin/python

import csv
import random

reader = csv.reader(open('../data/train.csv','rb'))
header = reader.next() # save header

rows = list(reader)

random.seed(1359)
random.shuffle(rows)

testSize = int(len(rows)*0.2)

writerTest = csv.writer(open('../data/test_xval.csv','wb'))
writerTrain = csv.writer(open('../data/train_xval.csv','wb'))

writerTest.writerow(header)
writerTest.writerows(rows[:testSize])

writerTrain.writerow(header)
writerTrain.writerows(rows[testSize:])
