# VarDial2021 - Romanian Dialect Identification (RDI)

This folder contains four files:

1. **shallow_models.py**: this Python3 script loads the tweets training data for the shared task, ```'dev.txt'```, and evaluates a Multinomial Naive Bayes model trained on TFIDF-transformed character ngrams. A Linear SVM can also be selected as the model to train.

2. **news+tweets_CNN.ipynb**: this Jupyter Notebook loads a CNN model trained on the news training dataset, and performs additional training on a tweets dataset. 

3. **trained_CNN_model**: this file contains the parameters used to train a CNN model on a news training dataset (from Ceolin and Zhang 2020).

4. **tweets_CNN.ipynb**: this Jupyter Notebook trains a CNN model directly on the tweets dataset, ```'dev.txt'```, and evaluates it.

References:

> Ceolin, A. & Zhang, H. (2020). *Discriminating between standard Romanian and Moldavian tweets using filtered character ngrams*. In Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects, 265-272.

