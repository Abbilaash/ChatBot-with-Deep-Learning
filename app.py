# Importing the needed modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random
import json
import tflearn

# Initializing the stemmer
stemmer = LancasterStemmer()

# Loading the training data from intents.json
with open('intents.json') as file:
    data = json.load(file)

# Extracting the data and splitting the data
words = []
labels = []
docs_x = []    # list of all the patterns
docs_y = []    # list of all the tags for each pattern in docs_x

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern) # For each pattern we will turn it into a list of words
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])






