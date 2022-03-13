# This file defines a number of utility functions for similarity matching，
# which are used in intent matching and question answering.

import re
import numpy as np

from math import log
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# Tokenisation and pre-processing of the corpus.
def get_tokenized_corpus(corpus):
    # put the tokenized corpus in a list
    tokenized_corpus = []
    stop_words = stopwords.words('english')
    sb_stemmer = SnowballStemmer('english')
    for sentence in corpus:
        sentence = sentence.lower()
        sentence = re.sub(r'[^ a-z0-9]', '', sentence)
        tokenized_sentence = []
        for word in word_tokenize(sentence):
            word = sb_stemmer.stem(word)
            if word not in stop_words:
                tokenized_sentence.append(word)
        tokenized_corpus.append(tokenized_sentence)
    return tokenized_corpus


# Get the non-repeating set of words in the corpus.
def get_vocabulary(tokenized_corpus):
    vocabulary = []
    for sentence in tokenized_corpus:
        for word in sentence:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


# Acquisition of bag-of-words models of the corpus.
def get_bow(vocabulary, tokenized_corpus):
    bow = []
    for idx_sent, sentence in enumerate(tokenized_corpus):
        vector = np.zeros(len(vocabulary))
        for word in sentence:
            idx_word = vocabulary.index(word)
            vector[idx_word] += 1
        bow.append(vector)
    return bow


# Perform tf-idf weighting on the input bag-of-words model.
def get_tfidf_bow(bow, vocabulary, tokenized_corpus):
    weighted_bow = []
    for vector in bow:
        weighted_vector = []
        for idx_freq, frequency in enumerate(vector):
            tf = log(1 + frequency)
            idf = 1
            '''
            ' This section is commented to speed up the computing.
            ' Doing so makes the program performing only log frequency term weighting.
            ' Uncomment to use TF-IDF weighting, but it is extremely slow.

            word = vocabulary[idx_freq]
            n = 0  # number of documents containing the word
            for document in tokenized_corpus:
                if word in document:
                    n += 1
            idf = log(len(tokenized_corpus) / n)
            '''
            tf_idf = tf * idf + 0.01  # 0.01 is for smoothing
            weighted_vector.append(tf_idf)
        weighted_bow.append(weighted_vector)
    return weighted_bow
