import re
import random
import string

import pandas as pd
from joblib import dump, load
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer


# Create a snowball stemmer analyzer
def stemmed_words(doc):
    sb_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (sb_stemmer.stem(w) for w in analyzer(doc))


# Train a GDBT classifier that classifies the intent of the input.
def train_classifier():
    # Reading the database and pre-processing.
    filepath = "datasets/Small_Talk_Dataset.csv"
    data = pd.read_csv(filepath, header=None).values
    X, y = data[1:, 0], data[1:, 1]
    tran_punc = str.maketrans({key: None for key in string.punctuation})
    X = [x.translate(tran_punc).lower() for x in X]

    # Training the classifier.
    count_vector = CountVectorizer(analyzer=stemmed_words)
    X_train_counts = count_vector.fit_transform(X)
    classifier = GradientBoostingClassifier(n_estimators=200).fit(X_train_counts, y)

    # Save the classifier to a file for further use.
    dump(count_vector, 'models/smallTalkVector.joblib')
    dump(classifier, 'models/smallTalkClassifier.joblib')
    return count_vector, classifier


# Use templates to respond to user input.
def response(sentence, user_name, bot_name):
    # Use a pre-trained model to accelerate computing.
    try:
        count_vector = load('models/smallTalkVector.joblib')
        classifier = load('models/smallTalkClassifier.joblib')
    except:
        count_vector, classifier = train_classifier()

    # Pre-processing of the input.
    sentence = sentence.lower()
    sentence = re.sub(r'[^ a-z]', '', sentence)

    # Use the trained model to determine the category of the user input.
    new_data_counts = count_vector.transform([sentence])
    predicted = classifier.predict(new_data_counts)[0]

    # Define the database of responses.
    basic_greetings_intros = ['Hi', 'Hello']
    advanced_greetings_intros = ['Not bad', 'Pretty good', 'I\'m doing well']
    advanced_greetings_outros = [', thanks', ', thank you', ', cheers', '']
    user_name_intros = ['You are', 'Your name is']
    bot_name_intros = ['I am', 'My name is', 'Call me']
    weather_intros = ['I guess', 'I suppose']
    weather_outros = ['Sunny', 'Rainy', 'Cloudy', 'Windy']

    # Respond according to the database and reply templates.
    if predicted == 'basic_greetings':
        intro = random.sample(basic_greetings_intros, 1)[0]
        if user_name == ' ':
            reply = '[{}]: {}!'.format(bot_name, intro)
        else:
            reply = '[{}]: {} {}!'.format(bot_name, intro, user_name)
    elif predicted == 'advanced_greetings':
        intro = random.sample(advanced_greetings_intros, 1)[0]
        outro = random.sample(advanced_greetings_outros, 1)[0]
        reply = '[{}]: {}{}.'.format(bot_name, intro, outro)
    elif predicted == 'identity_user':
        intro = random.sample(user_name_intros, 1)[0]
        if user_name == ' ':
            reply = '[{}]: Sorry, I don\'t know your name.'.format(bot_name)
        else:
            reply = '[{}]: {} {}.'.format(bot_name, intro, user_name)
    elif predicted == 'identity_bot':
        intro = random.sample(bot_name_intros, 1)[0]
        reply = '[{}]: {} {}.'.format(bot_name, intro, bot_name)
    elif predicted == 'weather':
        intro = random.sample(weather_intros, 1)[0]
        outro = random.sample(weather_outros, 1)[0]
        reply = '[{}]: I don\'t know, but {} it\'s {}.'.format(bot_name, intro, outro)
    else:
        reply = '[{}]: Sorry, I did not get your point.'.format(bot_name)

    return reply
