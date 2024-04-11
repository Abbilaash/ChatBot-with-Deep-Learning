# Importing the needed modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow,tflearn
import random,json,numpy

# Initializing the stemmer
stemmer = LancasterStemmer()

# Loading the training data from intents.json
with open('intents.json') as file:
    data = json.load(file)

# Extracting the data and splitting the data
# docs_x will contain all the patterns
# docs_y is a list of all the tags for each pattern in docs_x
words,labels,docs_x,docs_y = [],[],[],[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern) # For each pattern we will turn it into a list of words
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stemming the words -> creating the stemmed volabulary
words = [stemmer.stem(w.lower()) for w in words if w not in ['?','!']]
words = sorted(list(set(words)))
labels = sorted(labels)






