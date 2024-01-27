# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# to stem words
from nltk.stem import PorterStemmer

# create an instance of class PorterStemmer
stemmer = PorterStemmer()

# importing json lib
import json
import pickle
import numpy as np

words = []  # list of unique roots words in the data
classes = []  # list of unique tags in the data
pattern_word_tags_list = []  # list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

# words to be ignored while creating Dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# open the JSON file, load data from it.
with open('PRO-C119-Project-Boilerplate-main/intents.json') as train_data_file:
    data = json.load(train_data_file)

# creating function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  #
    return stem_words

for intent in data['intents']:
    # Add all words of patterns to list
    for pattern in intent['patterns']:
        pattern_word = nltk.word_tokenize(pattern)
        words.extend(pattern_word)
        pattern_word_tags_list.append((pattern_word, intent['tag']))

    # Add all tags to the classes list
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(pattern_word_tags_list[0])
print(classes)

def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        for pattern in intent['patterns']:

            pattern_words = nltk.word_tokenize(pattern)
            pattern_words = [w for w in pattern_words if w not in ignore_words]

            pattern_word_tags_list.append((pattern_words, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words)
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    return stem_words, classes, pattern_word_tags_list

# Training Dataset:
# Input Text----> as Bag of Words 
# Tags-----------> as Label
training_data = []
no_of_tags = len(classes)
labels = [0] * no_of_tags

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        # example: word_tags = (['hi', 'there'], 'greetings']
        pattern_words = word_tags[0]  # ['Hi' , 'There]
        bag_of_words = []

        # stemming pattern words before creating Bag of words
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        for word in stem_words:
            if word in stemmed_pattern_word:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        bag.append(bag_of_words)

    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []

    for word_tags in pattern_word_tags_list:

        # Start with list of 0s
        labels_encoding = list([0] * len(classes))  #

        # example: word_tags = (['hi', 'there'], 'greetings']

        tag = word_tags[1]  # 'greetings'

        tag_index = classes.index(tag)

        # Labels Encoding
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)

    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)

    with open("words.pkl", "wb") as stem_words_file, open("classes.pkl", "wb") as classes_file:
        stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
        train_x = bag_of_words_encoding(stem_words, word_tags_list)
        train_y = class_label_encoding(tag_classes, word_tags_list)
    return train_x, train_y

bow_data, label_data = preprocess_train_data()

# after completing the code, remove comment from print statements
print("first BOW encoding: ", bow_data[0])
print("first Label encoding: ", label_data[0])