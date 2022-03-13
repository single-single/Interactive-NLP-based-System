import nltk

from joblib import dump, load

from choiceClassifier import choice_classifier
from intentMatching import matching
from identityManagement import extract_name
from smallTalk import response
from questionAnswering import retrieve, update_database
from transactions import transaction
from gamePlaying import game


# Download required packages.
nltk.download('stopwords')
nltk.download('punkt')


# Defining the skeleton of the chatbot.
class Chatbot:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __getstate__(self):
        return self.data

    def __setstate__(self, data):
        self.data = data

    # Retrieves information about the corresponding user in the chat data.
    # This function is for further development and is not used in this coursework.
    def retrieve_data(self, name):
        for k, d in enumerate(self.data):
            if d['name'] == name:
                return d

    # Updates information about the corresponding user in the chat data.
    # This function is for further development and is not used in this coursework.
    def update_data(self, name, term, new_data):
        for k, d in enumerate(self.data):
            if d['name'] == name:
                d[term] = new_data
                return

    # Quit the chat and save the user's information to a file if they have provided their name.
    def quit(self, is_remember):
        if is_remember:
            print('[%s]: Our chat data is saved in a file named "chatData.joblib".' % self.name)
            dump(self.data, 'datasets/chatData.joblib')
        print('[%s]: Goodbye.' % self.name)

    # The main function for chat.
    def chat(self):
        bot_name = self.name
        user_name = ' '
        print('[%s]: Hello, I am your AI assistant. My name is %s.' % (bot_name, bot_name))

        is_new_task = True      # This variable indicates whether a new session needs to be opened.
        is_remember = False     # This variable indicates whether the user has provided a name.
        attempts = 0            # The number of times the bot failed to understand the user's intent.

        # The conversation continues until the user wants to exit.
        while True:
            # Get user instructions.
            # The prompt changes according to the value of 'is_new_task'.
            if is_new_task:
                attempts = 0
                reply = input('[%s]: How Can I help you?\n[You]: ' % bot_name).strip()
            else:
                attempts += 1
                is_new_task = True
                reply = input('[%s]: Could you please describe it more clearly?\n[You]: ' % bot_name).strip()

            # Handle empty input.
            while reply == '':
                reply = input('[%s]: Please input something.\n[You]: ' % bot_name)

            # Use the 'matching' function to map the user's input to the intent.
            intent = matching(reply)

            # Remember the name of the user.
            if intent == 'identity':
                user_name = extract_name(reply)
                choice = input('[%s]: Is your name %s? [y/n]\n[You]: ' % (bot_name, user_name))
                # If the name is extracted incorrectly, the user is asked to enter his or her name.
                if not choice_classifier(choice, bot_name):
                    user_name = input('[%s]: Please tell me your name.\n[You]: My name is: ')
                is_remember = True
                # The greeting is based on whether or not the user's data has been saved previously.
                if not self.retrieve_data(user_name):
                    print('[%s]: Nice to meet you, %s.' % (bot_name, user_name))
                    d = {'name': user_name}
                    self.data.append(d)
                else:
                    print('[%s]: Hi, %s. How I miss you.' % (bot_name, user_name))
            # Have a small talk with the user.
            elif intent == 'talk':
                answer = response(reply, user_name, bot_name)
                print(answer)
            # Answering users' questions.
            elif intent == 'answering':
                print('[%s]: Let me access my database...' % bot_name)
                answers = retrieve(reply)
                if not answers:
                    print('[%s]: Sorry, I am not yet able to answer this question.' % bot_name)
                    continue
                # Try the responses returned by the 'retrieve' function in order.
                # If positive feedback is received, store this response in the database.
                for i, answer in enumerate(answers):
                    print('[{}]: {}'.format(bot_name, answer))
                    choice = input('[%s]: Did this answer your question? [y/n]\n[You]: ' % bot_name)
                    if not choice_classifier(choice, bot_name):
                        if i == len(answers)-1:
                            print('[%s]: Sorry, I am not yet able to answer this question.' % bot_name)
                        else:
                            print('[%s]: Let me try again.' % bot_name)
                        continue
                    else:
                        update_database(reply, answer)
                        break
            # Open the restaurant reservation system.
            elif intent == 'transaction':
                choice = input('[%s]: Would you like to book a restaurant table? [y/n]\n[You]: ' % bot_name)
                if choice_classifier(choice, bot_name):
                    transaction(bot_name)
            # Play a little game with the user.
            elif intent == 'game':
                choice = input('[%s]: Would you like to play a game? [y/n]\n[You]: ' % bot_name)
                if choice_classifier(choice, bot_name):
                    game(bot_name)
            # Exit the chat.
            elif intent == 'quit':
                choice = input('[%s]: Do you want to exit the chat? [y/n]\n[You]: ' % bot_name)
                if choice_classifier(choice, bot_name):
                    self.quit(is_remember)
                    break
            # If the user's instructions cannot be understood.
            else:
                # 2 attempts to reinterpret the user's instructions.
                if attempts < 2:
                    is_new_task = False
                print('[%s]: Sorry, I am unable to understand your instruction.' % bot_name)


# Attempt to load saved user data from a file.
try:
    chat_data = load('datasets/chatData.joblib')
except:
    chat_data = []

# Start a chat.
chatbot = Chatbot('Sophia', chat_data)
chatbot.chat()
