import nltk
nltk.download('wordnet')
nltk.download('punkt')  # Untuk tokenisasi
nltk.download('omw-1.4')  # Jika menggunakan lemmatizer berbasis WordNet
from nltk.stem import WordNetLemmatizer
from flask import Flask, jsonify, request, render_template

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random

from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv

# Muat file .env untuk API key
load_dotenv()
API_KEY = os.getenv("API_KEY")  # Ambil API key dari environment variables

# Inisialisasi model, data, dan Flask
model = load_model('timahtechbot_model.h5')
intents = json.loads(open('data_panduan_formatted.json', encoding="utf8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

app = Flask(__name__)

# Membersihkan kalimat input pengguna
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Mengubah kalimat menjadi bag of words
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

# Prediksi kelas kalimat menggunakan model
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

# Mendapatkan respons chatbot berdasarkan intent
def getResponse(ints, intents_json):
    if not ints:
        return "Sorry, I didn't understand that."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Mendapatkan respons chatbot
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Fungsi untuk memeriksa API key
def check_api_key():
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        return jsonify({"message": "Unauthorized"}), 401
    return None

@app.route("/", methods=['GET'])
def hello():
    return jsonify({"message": "Welcome to Timah TechBot API!"})

# Tambahkan route baru di sini untuk UI
@app.route("/ui")
def chat_ui():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    # Memeriksa API key
    unauthorized_response = check_api_key()
    if unauthorized_response:
        return unauthorized_response

    # Ambil pesan dari body permintaan
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"message": "No message provided"}), 400

    # Dapatkan respons dari chatbot
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
