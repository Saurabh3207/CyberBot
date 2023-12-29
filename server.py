# backend/app.py
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
import random
import pickle
import nltk
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the words, classes, and intents
data = pickle.load(open("training_data.pkl", "rb"))
words = data['words']
classes = data['classes']
intents = data['intents']

# Load the trained model
model = load_model('model.keras')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json['user_input']
        if not user_input:
            raise ValueError("User input is empty")

        predictions = predict_class(user_input, model)
        if predictions:
            intent_tag = predictions[0]['intent']
            for intent_data in intents['intents']:
                if intent_data['tag'] == intent_tag:
                    responses = intent_data['responses']

            bot_response = random.choice(responses)
            return jsonify({"bot_response": bot_response})

        return jsonify({"bot_response": "I'm sorry, but I didn't understand that."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000)
