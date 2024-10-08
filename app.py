import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import json
import random
import requests
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Muat file .env untuk API key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Load model dan file lainnya
model = load_model('chatbot_model.h5')
intents = json.loads(open('data_panduan_formatted.json', encoding="utf8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()

# Simpan status percakapan pengguna
user_status = {}

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
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Mendapatkan respons chatbot berdasarkan intent
def getResponse(ints, intents_json):
    if not ints:
        return "Maaf, saya tidak memahami itu."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Fungsi untuk memeriksa API key
def check_api_key():
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        return jsonify({"message": "Unauthorized"}), 401
    return None

# Fungsi untuk membuat tiket melalui backend API
def create_ticket(user_details, user_id, title, description):
    try:
        url = "https://your-backend-api.com/ticket"  # Ganti dengan URL API backend Anda
        payload = {
            "user_id": user_id,
            "title": title,
            "description": description
        }
        response = requests.post(url, json=payload)

        if response.status_code == 201:
            return "Tiket berhasil dibuat!"
        else:
            return f"Terjadi kesalahan saat membuat tiket: {response.status_code} - {response.json().get('message', 'Unknown Error')}"
    except requests.ConnectionError:
        return "Tidak bisa menghubungi server. Silakan coba lagi nanti."
    except Exception as e:
        return f"Gagal membuat tiket: {str(e)}"

# Fungsi untuk melihat tiket melalui backend API
def view_tickets(user_id):
    try:
        url = f"https://your-backend-api.com/tickets/{user_id}"  # Ganti dengan URL API backend Anda
        response = requests.get(url)

        if response.status_code == 200:
            tickets = response.json()
            if tickets:
                ticket_list = []
                for ticket in tickets:
                    ticket_list.append(f"Tiket ID: {ticket['ticket_id']}, Judul: {ticket['title']}, Status: {ticket['status']}")
                return "\n".join(ticket_list)
            else:
                return "Anda belum memiliki tiket."
        else:
            return f"Gagal mengambil tiket: {response.status_code}"
    except Exception as e:
        return f"Gagal menghubungi server: {str(e)}"

# Mendapatkan respons chatbot
def chatbot_response(msg, user_id=None):
    try:
        # Cek apakah pengguna sedang dalam proses pembuatan tiket
        if user_id in user_status and user_status[user_id].get('intent') == 'buat_tiket':
            if 'title' not in user_status[user_id]:
                user_status[user_id]['title'] = msg
                return "Silakan berikan detail masalah Anda."
            elif 'description' not in user_status[user_id]:
                user_status[user_id]['description'] = msg
                # Buat tiket setelah mendapatkan judul dan deskripsi
                response = create_ticket(user_details=user_status[user_id]['description'],
                                         user_id=user_id,
                                         title=user_status[user_id]['title'],
                                         description=user_status[user_id]['description'])
                user_status.pop(user_id)  # Hapus status setelah tiket dibuat
                return response

        # Prediksi intent berdasarkan pesan pengguna
        ints = predict_class(msg, model)

        if ints:
            intent = ints[0]['intent']
            
            # Jika intent adalah "buat_tiket"
            if intent == "buat_tiket":
                user_status[user_id] = {'intent': 'buat_tiket'}
                return "Silakan masukkan judul untuk tiket Anda."
            
            # Jika intent adalah "lihat_tiket"
            elif intent == "lihat_tiket":
                response = view_tickets(user_id)
            
            else:
                response = getResponse(ints, intents)
        else:
            response = "Maaf, saya tidak mengerti."
    
    except Exception as e:
        response = f"Terjadi kesalahan: {str(e)}"

    return response

# Route untuk halaman depan
@app.route("/", methods=['GET'])
def hello():
    return jsonify({"message": "Welcome to Timah TechBot API!"})

# Route untuk UI admin chat
@app.route("/admin/roomchat")
def chat_ui():
    return render_template("index.html")

# Route untuk obrolan chatbot
@app.route('/chat', methods=['POST'])
def chat():
    unauthorized_response = check_api_key()
    if unauthorized_response:
        return unauthorized_response

    data = request.get_json()
    user_message = data.get("message")
    user_id = data.get("user_id")

    if not user_message:
        return jsonify({"message": "No message provided"}), 400

    if not user_id:
        return jsonify({"message": "User ID not provided"}), 400

    # Pastikan bahwa response bukan 'undefined'
    response = chatbot_response(user_message, user_id)
    
    # Cek apakah response valid dan bukan 'undefined'
    if response is None or response == "":
        response = "Maaf, terjadi kesalahan dalam memproses permintaan Anda."

    return jsonify({"response": response})

# Route untuk membuat tiket
@app.route('/create_ticket', methods=['POST'])
def create_ticket_route():
    unauthorized_response = check_api_key()
    if unauthorized_response:
        return unauthorized_response

    data = request.get_json()
    user_id = data.get('user_id')
    title = data.get('title')
    description = data.get('description')

    if not user_id or not title or not description:
        return jsonify({"message": "Data tidak lengkap"}), 400

    response = create_ticket(user_details=description, user_id=user_id, title=title, description=description)
    return jsonify({"message": response})

# Route untuk melihat tiket
@app.route('/view_tickets', methods=['GET'])
def view_tickets_route():
    unauthorized_response = check_api_key()
    if unauthorized_response:
        return unauthorized_response

    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"message": "User ID tidak diberikan"}), 400

    response = view_tickets(user_id)
    return jsonify({"tickets": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
