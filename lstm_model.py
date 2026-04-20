import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# 📥 LOAD DATA
# =========================
df = pd.read_csv('data/data_clean.csv')

# =========================
# ⚖️ BALANCING DATA
# =========================
df_neg = df[df['sentimen'] == 'negatif']
df_pos = df[df['sentimen'] == 'positif']
df_net = df[df['sentimen'] == 'netral']

max_len_data = max(len(df_neg), len(df_pos), len(df_net))

df_neg = resample(df_neg, replace=True, n_samples=max_len_data, random_state=42)
df_pos = resample(df_pos, replace=True, n_samples=max_len_data, random_state=42)
df_net = resample(df_net, replace=True, n_samples=max_len_data, random_state=42)

df = pd.concat([df_neg, df_pos, df_net])

print("Distribusi setelah balancing:")
print(df['sentimen'].value_counts())

# =========================
# 🧠 TOKENIZER
# =========================
max_words = 5000
max_len = 120

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean'])

X = tokenizer.texts_to_sequences(df['clean'])
X = pad_sequences(X, maxlen=max_len, padding='post')

# =========================
# 🎯 LABEL ENCODER
# =========================
le = LabelEncoder()
y = le.fit_transform(df['sentimen'])

# =========================
# ✂️ SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# ⚖️ CLASS WEIGHT
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =========================
# 🤖 MODEL LSTM UPGRADE
# =========================
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# =========================
# ⚙️ COMPILE
# =========================
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =========================
# 🚀 TRAINING
# =========================
print("\n🚀 Training model...")
model.fit(
    X_train,
    y_train,
    epochs=12,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# =========================
# 📊 EVALUASI
# =========================
print("\n📊 Evaluasi Model")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 💾 SIMPAN MODEL
# =========================
os.makedirs('model', exist_ok=True)

model.save('model/model_lstm.keras')

with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✅ Model berhasil disimpan!")