import re
import string

import pandas as pd
from joblib import dump, load
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

from choiceClassifier import choice_classifier


# Create a snowball stemmer analyzer
def stemmed_words(doc):
    sb_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (sb_stemmer.stem(w) for w in analyzer(doc))


# Train a KNN classifier that classifies the intent of the input.
def train_classifier():
    filepath = "datasets/Transactions_Dataset.csv"
    data = pd.read_csv(filepath, header=None).values
    X, y = data[1:, 0], data[1:, 1]
    tran_punc = str.maketrans({key: None for key in string.punctuation})
    X = [x.translate(tran_punc).lower() for x in X]

    count_vector = CountVectorizer(analyzer=stemmed_words)
    X_train_counts = count_vector.fit_transform(X)
    classifier = KNeighborsClassifier(n_neighbors=1).fit(X_train_counts, y)

    dump(count_vector, 'models/transactionsVector.joblib')
    dump(classifier, 'models/transactionsClassifier.joblib')
    return count_vector, classifier


# Get the restaurant the user wants to book.
def get_restaurant(count_vector, classifier, restaurants, bot_name):
    intent = input('[%s]: There is one Chinese restaurant and one Thai restaurant available, '
                   'which one would you like to book?\n[You]: ' % bot_name)
    while True:
        # Pre-processing and using the model to determine the category of the user input.
        intent = intent.lower()
        intent = re.sub(r'[^ a-z]', '', intent)
        new_data_counts = count_vector.transform([intent])
        predicted = classifier.predict(new_data_counts)[0]

        # Respond to user input and extract the restaurant name.
        word_counter = 0
        if predicted == 'restaurant':
            for word in word_tokenize(intent):
                if word in restaurants:
                    restaurant = word
                    word_counter += 1
            if word_counter != 1:
                intent = input('[%s]: Sorry, please specify the restaurant you want to book.\n[You]: ' % bot_name)
            else:
                break
        else:
            intent = input('[%s]: Sorry, I can\'t understand your input, please try again.\n[You]: ' % bot_name)
    return restaurant


# Get the date the user wants to book.
def get_date(count_vector, classifier, months, bot_name):
    intent = input('[%s]: What date would you like to reserve your place?\n[You]: ' % bot_name)
    while True:
        # Pre-processing and using the model to determine the category of the user input.
        intent = intent.lower()
        intent = re.sub(r'[^ a-z0-9]', '', intent)
        new_data_counts = count_vector.transform([intent])
        predicted = classifier.predict(new_data_counts)[0]

        # Respond to user input and extract the date.
        word_counter = 0
        if predicted == 'date':
            for word in word_tokenize(intent):
                if word in months:
                    month = word.capitalize()
                    word_counter += 1
            day = re.sub(r'[^ 0-9]', '', intent)
            day = day.strip()
            if (' ' in day) or (day == '') or (word_counter != 1):
                intent = input('[%s]: Sorry, please specify the precise date of your reservation.\n[You]: ' % bot_name)
            elif not 1 <= int(day) <= 31:
                intent = input('[%s]: Sorry, please provide a valid date of your reservation.\n[You]: ' % bot_name)
            else:
                choice = input('[{}]: Are you looking to book a place on {} {}? [y/n]\n[You]: '.format(bot_name, month, day))
                if choice_classifier(choice, bot_name):
                    break
                else:
                    intent = input('[%s]: Could you please re-enter the date you want to book?\n[You]: ' % bot_name)
        else:
            intent = input('[%s]: Sorry, I can\'t understand your input, please try again.\n[You]: ' % bot_name)
    return day, month


# Get whether the user wants to book lunch or dinner.
def get_meal(count_vector, classifier, types, bot_name):
    intent = input('[%s]: Would you like to book lunch or dinner?\n[You]: ' % bot_name)
    while True:
        # Pre-processing and using the model to determine the category of the user input.
        intent = intent.lower()
        intent = re.sub(r'[^ a-z]', '', intent)
        new_data_counts = count_vector.transform([intent])
        predicted = classifier.predict(new_data_counts)[0]

        # Respond to user input and extract the meal type.
        word_counter = 0
        if predicted == 'type':
            for word in word_tokenize(intent):
                if word in types:
                    meal = word
                    if meal == 'supper':
                        meal = 'dinner'
                    word_counter += 1
            if word_counter != 1:
                intent = input('[%s]: Sorry, please specify whether you want to book lunch or dinner.\n[You]: ' % bot_name)
            else:
                break
        else:
            intent = input('[%s]: Sorry, I can\'t understand your input, please try again.\n[You]: ' % bot_name)
    return meal


# Get the number of people coming to the meal.
def get_people(bot_name):
    intent = input('[%s]: How many people do you have?\n[You]: ' % bot_name)
    # Respond to user input and extract the number of people.
    while True:
        people = re.sub(r'[^ 0-9]', '', intent)
        people = people.strip()
        if (' ' in people) or (people == ''):
            intent = input('[%s]: Sorry, I can\'t understand your input, please try again.\n[You]: ' % bot_name)
        elif not 0 < int(people) < 100:
            intent = input('[%s]: Sorry, please provide a valid number of people.\n[You]: ' % bot_name)
        else:
            break
    return int(people)


# Open a dialogue for restaurant reservations.
# The system reserves place for you in order of
# date, lunch or dinner, and number of people.
def transaction(bot_name):
    # Use a pre-trained model to accelerate computing.
    try:
        count_vector = load('models/transactionsVector.joblib')
        classifier = load('models/transactionsClassifier.joblib')
    except:
        count_vector, classifier = train_classifier()

    # Store available restaurant reservations in a list.
    # Each space is stored in a tuple of the form (Day, Month, Lunch or dinner, table size).
    spaces_c = [('22', 'December', 'lunch', 's'), ('23', 'December', 'lunch', 's'), ('23', 'December', 'dinner', 's'),
              ('24', 'December', 'lunch', 's'), ('25', 'December', 'lunch', 's'), ('25', 'December', 'dinner', 's'),
              ('26', 'December', 'lunch', 's'), ('22', 'December', 'lunch', 'm'), ('22', 'December', 'dinner', 'm'),
              ('24', 'December', 'lunch', 'm'), ('25', 'December', 'lunch', 'm'), ('25', 'December', 'dinner', 'm'),
              ('23', 'December', 'lunch', 'l'), ('24', 'December', 'lunch', 'l'), ('25', 'December', 'dinner', 'l')]
    spaces_t = [('22', 'December', 'lunch', 's'), ('22', 'December', 'dinner', 's'), ('23', 'December', 'lunch', 's'),
              ('24', 'December', 'lunch', 's'), ('24', 'December', 'dinner', 's'), ('25', 'December', 'lunch', 's'),
              ('26', 'December', 'lunch', 's'), ('22', 'December', 'dinner', 'm'), ('23', 'December', 'lunch', 'm'),
              ('24', 'December', 'lunch', 'm'), ('24', 'December', 'dinner', 'm'), ('25', 'December', 'lunch', 'm'),
              ('22', 'December', 'lunch', 'l'), ('23', 'December', 'dinner', 'l'), ('25', 'December', 'lunch', 'l')]

    restaurants = ['chinese', 'thai']
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
              'august', 'september', 'october', 'november', 'december']
    types = ['lunch', 'dinner', 'supper']

    print('[%s]: Hello, welcome to the restaurant booking system.' % bot_name)

    # Get the information needed to reserve a restaurant table.
    restaurant = get_restaurant(count_vector, classifier, restaurants, bot_name)
    day, month = get_date(count_vector, classifier, months, bot_name)
    meal = get_meal(count_vector, classifier, types, bot_name)
    people = get_people(bot_name)

    # Determine the type of table the user needs based on the number of people.
    if people < 4:
        table = 's'
    elif 4 <= people < 8:
        table = 'm'
    elif 8 <= people < 12:
        table = 'l'
    else:
        print('[%s]: Sorry, we don\'t have space for that many people. ' % bot_name)
        return

    # Reply whether the booking was successful or not based on
    # the information extracted and the availability of spaces in the database.
    if restaurant == 'chinese':
        spaces = spaces_c
    else:
        spaces = spaces_t
    book_info = (day, month, meal, table)
    if book_info in spaces:
        print('[{}]: A successful {} booking has been made for you on {} {} for {} people.'.format(bot_name, meal, month, day, people))
    else:
        print('[%s]: Sorry, no suitable space was found for you, and the reservation failed.' % bot_name)
