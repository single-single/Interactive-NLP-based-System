import difflib

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

from toolkit import get_tokenized_corpus, get_vocabulary, get_bow, get_tfidf_bow


# If there is no data similar to the input in the database,
# the data is appended at the end of the database.
# Note: Since this program writes data after EOF,
# manually modifying the database may result in write errors.
# Hence, please try not to edit the database manually.
def update_database(query, reply):
    filepath = "datasets/Question_Answering_Dataset.csv"
    data = pd.read_csv(filepath, header=None).values
    X, y = data[1:, 1], data[1:, 2]
    query, reply = query.lower(), reply.lower()

    is_dup_x, is_dup_y = False, False
    for q in X:
        if difflib.SequenceMatcher(None, query, q).ratio() > 0.9:
            is_dup_x = True
    for r in y:
        if difflib.SequenceMatcher(None, reply, r).ratio() > 0.9:
            is_dup_y = True

    if not (is_dup_x and is_dup_y):
        data_frame = pd.DataFrame({"Question": query, "Answer": reply}, index=[len(X)+1])
        data_frame.to_csv(filepath, index=True,  mode='a', header=False)


# Search for similar questions from the database and return the corresponding answers.
def retrieve(sentence):
    # Reading the database.
    filepath = "datasets/Question_Answering_Dataset.csv"
    data = pd.read_csv(filepath, header=None).values
    X, y = data[1:, 1], data[1:, 2]

    # Get weighted bag-of-words models of the database.
    tokenized_corpus = get_tokenized_corpus(X)
    vocabulary = get_vocabulary(tokenized_corpus)
    bow = get_bow(vocabulary, tokenized_corpus)
    weighted_bow = get_tfidf_bow(bow, vocabulary, tokenized_corpus)

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
    vector_query = get_tfidf_bow([vector_query], vocabulary, tokenized_corpus)[0]

    # Calculate the cosine similarity of the input to each statement
    # in the database and rank them from highest to lowest.
    similarity_all = {}
    for idx_vec, document in enumerate(weighted_bow):
        similarity = dot(vector_query, document)/(norm(vector_query)*norm(document))
        question = X[idx_vec]
        similarity_all[idx_vec] = (question, similarity)
    similarity_all = sorted(similarity_all.items(), key=lambda x: x[1][1], reverse=True)

    # Returns all responses to the most similar questions.
    # If no similar question is found, return an empty list.
    indices = [similarity_all[0]]
    threshold = 0.8
    answers = []
    if indices[0][1][1] > threshold:
        i = 1
        while similarity_all[i][1] == indices[0][1]:
            indices.append(similarity_all[i])
            i += 1
        for item in indices:
            answers.append(y[item[0]])

    return answers
