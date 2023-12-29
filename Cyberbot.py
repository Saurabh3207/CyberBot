import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json
import pickle

# Download NLTK data
nltk.download('punkt') 
nltk.download('wordnet')

# Load intents from a JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Extract data for training
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Pad the sequences to have a consistent length
max_sequence_length = max(len(seq[0]) for seq in training)
for i, seq in enumerate(training):
    padding = [0] * (max_sequence_length - len(seq[0]))
    training[i][0] += padding

# Shuffle the training data
random.shuffle(training)

# Split the features and labels
train_x = np.array([seq[0] for seq in training])
train_y = np.array([seq[1] for seq in training])

# Build the neural network model
model = Sequential()
model.add(Dense(8, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=1000, batch_size=8, verbose=1)

# Save the model
model.save('model.keras')

# Save the words, classes, and intents to use in the chat
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y, 'intents': intents}, open("training_data.pkl", "wb"))