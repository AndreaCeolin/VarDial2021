#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

'''Load Data'''
training = open('data/dev.txt').readlines()

'''Prepare training set'''
trainDialectLabels = []
trainSamples = []
for line in training:
    sample, label = line.split('\t')
    trainDialectLabels.append(label.rstrip().rstrip('\u202c'))
    trainSamples.append(sample.replace('$NE$', '').translate(line.maketrans('', '', string.punctuation+'0123456789')))

'''Prepare test set'''
X_train = trainSamples[len(trainSamples)//5:]
y_train = trainDialectLabels[len(trainSamples)//5:]
X_test = trainSamples[:len(trainSamples)//5]
y_test = trainDialectLabels[:len(trainSamples)//5]

'''Train and evaluate the models'''
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(5,8), sublinear_tf=True)
scaler = StandardScaler(with_mean=False)
X_train = vectorizer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
model = MultinomialNB()
#switch to LinearSVC() for Linear SVM
#model=LinearSVC()
model.fit(X_train, y_train)
X_test = vectorizer.transform(X_test)
X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred, average='macro'))

