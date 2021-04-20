#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

'''Load Data'''
data = open('Dravidianlanid-Vardial2021-train.tsv').readlines()

'''Set Seed'''
random.seed(10)
random.shuffle(data)

'''Prepare training set'''
data_labels = []
data_sentences = []
for line in data:
    if len(line.split('\t')) == 2:
        sample, label = line.split('\t')
        data_labels.append(label.rstrip())
        data_sentences.append(sample.translate(line.maketrans('', '', string.punctuation+'0123456789')))

print(Counter(data_labels))

'''Train/Test split 80:20'''
X_train = data_sentences[:13337]
y_train = data_labels[:13337]
X_test = data_sentences[13337:]
y_test = data_labels[13337:]

'''Data matrices'''
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(5,8), sublinear_tf=True)
scaler = StandardScaler(with_mean=False)
X_train = vectorizer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
#switch to LinearSVC() for Linear SVM
#model=LinearSVC()
model = MultinomialNB()
model.fit(X_train, y_train)
X_test = vectorizer.transform(X_test)
X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred, average='micro'))

