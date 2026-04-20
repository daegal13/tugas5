import pandas as pd
import re
import string

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('data/data_labeled.csv')

df = df[df['komentar'].notna()]

# =========================
# STOPWORDS RINGAN
# =========================
stopwords = [
    "yang","dan","di","ke","dari","ini","itu","untuk","dengan",
    "atau","juga","karena","jadi","saya","aku","kamu","dia"
]

# =========================
# CLEAN FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    
    # hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # hapus mention & hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # hapus angka
    text = re.sub(r'\d+', '', text)
    
    # hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    
    # hapus stopwords
    words = [word for word in words if word not in stopwords]
    
    return " ".join(words)

# =========================
# APPLY
# =========================
df['clean'] = df['komentar'].apply(clean_text)

# hapus kosong
df = df[df['clean'].str.strip() != ""]

# =========================
# SIMPAN
# =========================
df.to_csv('data/data_clean.csv', index=False)

print("✅ Preprocessing selesai!")
print("Jumlah data:", len(df))
print(df[['komentar','clean']].head())