from flask import Flask, render_template, request
import pickle
import re
import string
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = load_model('model/model_lstm.keras')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# =========================
# HISTORY (SIMPAN SEMENTARA)
# =========================
history_data = []

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# ROUTE
# =========================
@app.route('/', methods=['GET','POST'])
def index():
    hasil = ""
    warna = ""
    confidence = 0

    if request.method == 'POST':
        teks = request.form['teks']

        teks_bersih = clean_text(teks)

        seq = tokenizer.texts_to_sequences([teks_bersih])
        pad = pad_sequences(seq, maxlen=120)

        pred = model.predict(pad)[0]

        if pred[0] > 0.45:
            hasil = "negatif"
            warna = "red"
        elif pred[2] > 0.45:
            hasil = "positif"
            warna = "green"
        else:
            hasil = "netral"
            warna = "orange"

        confidence = round(max(pred) * 100, 2)

        # simpan history
        history_data.append({
            "teks": teks,
            "hasil": hasil,
            "confidence": confidence
        })

    # hitung grafik
    count_neg = sum(1 for h in history_data if h['hasil'] == 'negatif')
    count_net = sum(1 for h in history_data if h['hasil'] == 'netral')
    count_pos = sum(1 for h in history_data if h['hasil'] == 'positif')

    return render_template(
        'index.html',
        hasil=hasil,
        warna=warna,
        confidence=confidence,
        history=history_data[::-1],
        neg=count_neg,
        net=count_net,
        pos=count_pos
    )

if __name__ == '__main__':
    app.run(debug=True)