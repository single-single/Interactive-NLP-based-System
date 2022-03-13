import difflib

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from nltk import RegexpTokenizer

from toolkit import get_tokenized_corpus, get_vocabulary, get_bow


# This function matches the input to the sentences in the corpus one by one alphabetically,
# and returns the index of the sentence with the highest similarity.
def direct_matching(sentence, corpus):
    # Pre-processing of the input.
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = sentence.lower()
    sentence = " ".join(tokenizer.tokenize(sentence))

    # Obtain and sort a dictionary containing all sentences and corresponding similarities.
    similarity_all = {}
    for idx_ques, question in enumerate(corpus):
        question = question.lower()
        question = " ".join(tokenizer.tokenize(question))
        similarity = difflib.SequenceMatcher(None, sentence, question).ratio()
        similarity_all[idx_ques] = similarity
    similarity_all = sorted(similarity_all.items(), key=lambda x: x[1], reverse=True)

    # Return the index of the sentence with the highest similarity.
    # If a similar utterance is not found, -1 is returned.
    threshold = 0.7
    predicted = similarity_all[0]
    if predicted[1] > threshold:
        return predicted[0]
    else:
        return -1


# This function matches the input to the sentences in the corpus one by one,
# and returns the index of the sentence with the highest similarity.
# Plain text based matching will be done first, and if the match fails,
# another vector based match will be done.
def matching(sentence):
    # Reading the database.
    filepath = "datasets/Intent_Matching_Dataset.csv"
    data = pd.read_csv(filepath, header=None).values
    X, y = data[1:, 0], data[1:, 1]
    # As question answering data is at the front of the database and
    # the volume of data far exceeds that of the other categories,
    # we read the database in reverse order for better intent matching.
    X, y = X[::-1], y[::-1]

    # Since the BOW model performs poorly in some very short utterances,
    # these short but meaningful utterances are handled here.
    idx_direct = direct_matching(sentence, X)
    if idx_direct != -1:
        return y[idx_direct]

    # Get bag-of-words models of the database.
    tokenized_corpus = get_tokenized_corpus(X)
    vocabulary = get_vocabulary(tokenized_corpus)
    bow = get_bow(vocabulary, tokenized_corpus)

    # Here some of the functions for corpus can also work on the input query.
    tokenized_query = get_tokenized_corpus([sentence])[0]
    vector_query = np.zeros(len(vocabulary))
    # Compute the bag-of-words model for the input utterance.
    for word in tokenized_query:
        try:
            idx_word = vocabulary.index(word)
            vector_query[idx_word] += 1
        except:
            continue

    # Calculate the cosine similarity of the input to each statement
    # in the database and rank them from highest to lowest.
    similarity_all = {}
    np.seterr(all='ignore')
    for idx_vec, document in enumerate(bow):
        similarity = dot(vector_query, document) / (norm(vector_query) * norm(document))
        similarity_all[idx_vec] = similarity
    similarity_all = sorted(similarity_all.items(), key=lambda x: x[1], reverse=True)

    # Returns the category with the highest similarity.
    # If no similar statement is found, then return 'other'.
    threshold = 0.7
    item = similarity_all[0]
    if item[1] > threshold:
        predicted = y[item[0]]
    else:
        predicted = 'other'

    return predicted
