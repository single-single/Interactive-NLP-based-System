import re
import string

from time import sleep
from random import choice
from random import randint
from nltk import word_tokenize

from choiceClassifier import choice_classifier


# This function extracts the number from the user's input.
# If a number is not extracted or more than one number is extracted, the user is prompted to re-enter it.
def extract_number(sentence, bot_name):
    while True:
        sentence = re.sub(r'[^ 0-9]', '', sentence)
        sentence = sentence.strip()
        if (' ' in sentence) or (sentence == ''):
            sentence = input('[%s]: The input is invalid, please enter again.\n[You]: ' % bot_name)
        else:
            break
    num = int(sentence)
    return num


# This function extracts the game name from the user's input.
# If a name is not extracted or more than one name is extracted, the user is prompted to re-enter it.
def extract_game(sentence, bot_name):
    result_list = ['tic', 'dice']
    while True:
        tran_punc = str.maketrans({key: ' ' for key in string.punctuation})
        sentence = sentence.translate(tran_punc).lower()
        word_counter = 0
        for word in word_tokenize(sentence):
            if word in result_list:
                game_name = word
                word_counter += 1
        if word_counter != 1:
            sentence = input('[%s]: The input is invalid, please enter again.\n[You]: ' % bot_name)
        else:
            break
    return game_name


# This function asks for and starts the game that the user wants to play.
def game(bot_name):
    sentence = input('[%s]: I can currently play tic-tac-toe and dice game,'
                     ' which one would you like to play?\n[You]: ' % bot_name)
    game_name = extract_game(sentence, bot_name)
    # Launch the appropriate game.
    if game_name == 'tic':
        tic_tac_toe = TicTacToe(bot_name)
        tic_tac_toe.play()
    else:
        dice = Dice(bot_name)
        dice.play()


# Defining the game of tic-tac-toe.
class TicTacToe:
    def __init__(self, bot_name):
        self.bot_name = bot_name
        self.X = 'X'
        self.O = 'O'
        self.B = ' '
        self.board = [self.B] * 9
        self.grid = '''
                  %s | %s | %s
                -------------
                  %s | %s | %s
                -------------
                  %s | %s | %s
                   '''

    # Display the chessboard on the screen.
    # Positions not played are marked with a number.
    def put_grid(self):
        blank = self.B
        board = self.board.copy()
        for i in range(1, 10):
            if board[i - 1] == blank:
                board[i - 1] = str(i)
        print(self.grid % tuple(board))

    # This function returns a list of all available positions to play.
    def valid_moves(self):
        moves = []
        for i in range(1, 10):
            if self.board[i - 1] == self.B:
                moves.append(i)
        return moves

    # Get the position where the user wants to drop the piece.
    def user_move(self):
        bot_name = self.bot_name
        valid_moves = self.valid_moves()
        sentence = input('[%s]: Please enter the position you wish to play in.\n[You]: ' % bot_name)
        while True:
            position = extract_number(sentence, bot_name)
            if position not in range(1, 10):
                sentence = input('[%s]: Please enter a number between 1 and 9.\n[You]: ' % bot_name)
            elif position not in valid_moves:
                sentence = input(
                    '[%s]: There is already a pawn in this position, please enter again.\n[You]: ' % bot_name)
            else:
                break
        return position

    # Randomly select a valid position for the bot to play in.
    def bot_move(self):
        bot_name = self.bot_name
        print('[%s]: Let me think about it...' % bot_name)
        sleep(0.5)
        valid_moves = self.valid_moves()
        position = choice(valid_moves)
        return position

    # This function determines if the game is over, and if so, returns the winning player.
    def is_win(self):
        blank = self.B
        board = self.board
        player = blank
        if board[0] == board[1] == board[2] != blank:
            player = board[0]
        elif board[3] == board[4] == board[5] != blank:
            player = board[3]
        elif board[6] == board[7] == board[8] != blank:
            player = board[6]
        elif board[0] == board[3] == board[6] != blank:
            player = board[0]
        elif board[1] == board[4] == board[7] != blank:
            player = board[1]
        elif board[2] == board[5] == board[8] != blank:
            player = board[2]
        elif board[0] == board[4] == board[8] != blank:
            player = board[0]
        elif board[2] == board[4] == board[6] != blank:
            player = board[2]

        if (blank not in board) or (player != blank):
            return player
        else:
            return False

    # This function defines the main loop of the tic-tac-toe game and is where the game starts.
    def play(self):
        X = self.X
        O = self.O
        B = self.B
        board = self.board
        bot_name = self.bot_name

        print('[%s]: Welcome to the Tic Tac Toe game.' % bot_name)
        self.put_grid()
        turn = X
        # The user plays the first move.
        # The bot and the user take turns to play until the game is over.
        while not self.is_win():
            if turn == X:
                move = self.user_move()
                board[move - 1] = turn
                turn = O
            else:
                move = self.bot_move()
                board[move - 1] = turn
                turn = X
            self.put_grid()

        # Print the results of the game.
        winner = self.is_win()
        if winner == X:
            print('[%s]: Congratulations, you win.' % bot_name)
        elif winner == O:
            print('[%s]: Sorry, you failed.' % bot_name)
        elif winner == B:
            print('[%s]: It is a draw.' % bot_name)

        # Determine if the user wants another round.
        sentence = input('[%s]: Would you like to play again? [y/n]\n[You]: ' % bot_name)
        if choice_classifier(sentence, bot_name):
            tic_tac_toe = TicTacToe(bot_name)
            tic_tac_toe.play()


# This class defines the dice game.
# In this game, the user needs to place a bet first,
# and then guess whether the sum of the 5 dice results is greater than 15.
# If the guess is correct, the user gets the money of the bet amount,
# otherwise the user loses these money.
class Dice:
    def __init__(self, bot_name):
        self.bot_name = bot_name
        self.sides = 6
        self.quantity = 5
        self.money = 100

    # This function extracts the guess from the user's input.
    # If a guess is not extracted or more than one guess is extracted, the user is prompted to re-enter it.
    def extract_guess(self, sentence):
        bot_name = self.bot_name
        result_list = ['big', 'small', 'greater', 'less']
        while True:
            word_counter = 0
            sentence = sentence.lower()
            for word in word_tokenize(sentence):
                if word in result_list:
                    guess = word
                    word_counter += 1
            if word_counter != 1:
                sentence = input('[%s]: The input is invalid, please enter again.\n[You]: ' % bot_name)
            else:
                break
        if guess == 'big' or 'greater':
            return True
        else:
            return False

    # This function gets and returns the amount that the user wants to bet.
    def bet(self):
        bot_name = self.bot_name
        print('[%s]: You now have a total of £%d.' % (bot_name, self.money))
        sentence = input('[%s]: Please enter the amount you wish to bet.\n[You]: ' % bot_name)
        amount = extract_number(sentence, bot_name)
        while amount not in range(1, self.money + 1):
            sentence = input('[%s]: Please enter a number between 1 and %d.\n[You]: ' % (bot_name, self.money))
            amount = extract_number(sentence, bot_name)
        else:
            self.money -= amount
            print('[%s]: You bet £%d, and you now have £%d left.' % (bot_name, amount, self.money))
        return amount

    # This function rolls the dice n times and returns the sum of their values.
    def roll(self):
        sides = self.sides
        quantity = self.quantity
        bot_name = self.bot_name
        results = 0
        for i in range(quantity):
            print('[%s]: Rolling the dice %d...' % (bot_name, i+1))
            side = randint(1, sides)
            results += side
            sleep(0.5)
        return results

    # This function defines the main loop of the dice game and is where the game starts.
    def play(self):
        bot_name = self.bot_name
        print('[%s]: Welcome to the dice game.' % bot_name)
        while True:
            amount = self.bet()
            real = self.roll()
            sentence = input('[%s]: Please guess whether the sum of the results of the five dice'
                             ' is "greater than" or "less than or equal to" 15?\n[You]: ' % bot_name)
            guess = self.extract_guess(sentence)

            # Determine if the user's guess is correct.
            if (guess and real > 15) or (not guess and real <= 15):
                print('[%s]: Congratulations, you win £%d.' % (bot_name, amount))
                self.money += amount * 2
            else:
                print('[%s]: Sorry, you lose £%d.' % (bot_name, amount))

            # Determine if the user can and wants play another round.
            if self.money == 0:
                print('[%s]: You have run out of money, game over.' % bot_name)
                break
            else:
                sentence = input('[%s]: Would you like to play again? [y/n]\n[You]: ' % bot_name)
                if not choice_classifier(sentence, bot_name):
                    break
