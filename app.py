# Importing the needed modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow,tflearn
import random,json,numpy,os

# Initializing the stemmer
stemmer = LancasterStemmer()

def train_model():
    global model,words,labels,training,output,data
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

    training = []
    output = []

    # list of zeroes with length of labels[]
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    # Building the model
    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)


    # Training the model and storing in model.tflearn
    if os.path.exists("model.tflearn.meta"):
        model.load("model.tflearn")
    else:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")
    
train_model()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))

def train_bot():
    global training, output
    while True:
        tag = input("Enter tag: ")
        question = input("Enter question: ")
        if question.lower() == "quit":
            break
        answer = input("Intended answer: ")
        


        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

#chat()
train_bot()






