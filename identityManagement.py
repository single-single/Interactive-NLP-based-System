import difflib
import string

from nltk import word_tokenize


# This function extracts the name of the user from the user's input.
def extract_name(sentence):
    # Handle exception
    if sentence.strip().find(' ') == -1:
        return sentence

    # De-punctuate and tokenise sentences.
    tran_punc = str.maketrans({key: None for key in string.punctuation})
    sentence = sentence.translate(tran_punc)
    sequence = word_tokenize(sentence)

    # Generate bigram tokens for sentences.
    phrases = []
    for w in range(len(sequence)-1):
        phrases.append((sequence[w] + ' ' + sequence[w+1]))

    # Match the bigram tokenised sentences with the keywords in the list and extract the name.
    name_statements = ['name is', 'i am', 'call me']
    for s in name_statements:
        for t in phrases:
            if difflib.SequenceMatcher(None, s, t.lower()).ratio() > 0.7:
                return sentence.split(t)[-1].strip()

    # If no keyword is matched, the last word of the sentence is used as the name by default.
    return sentence.split()[-1]
