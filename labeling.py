import pandas as pd

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('data/data_raw.csv')

df = df[df['komentar'].notna()]

# =========================
# KAMUS SENTIMEN (UPGRADE)
# =========================
positif_words = [
    "aman","bagus","cepat","mudah","mantap","top","lancar",
    "terpercaya","nyaman","sukses","praktis","recommended",
    "oke","keren","helpful","puas","worth","good","mantul"
]

negatif_words = [
    "hilang","error","gagal","lambat","jelek","buruk",
    "penipuan","hack","dibobol","kecewa","ribet","parah",
    "lemot","ngelag","refund","tidak masuk","gk masuk",
    "scam","penipu","fraud","sangat buruk"
]

negasi_words = ["tidak","gak","nggak","bukan","ga"]

# =========================
# FUNGSI SENTIMEN
# =========================
def get_sentiment(text):
    text = str(text).lower()
    words = text.split()
    
    score = 0

    for i, word in enumerate(words):
        
        # cek negasi
        if word in negasi_words and i+1 < len(words):
            next_word = words[i+1]

            if next_word in positif_words:
                score -= 1
            elif next_word in negatif_words:
                score += 1
        
        # normal
        if word in positif_words:
            score += 1
        elif word in negatif_words:
            score -= 1

    # =========================
    # THRESHOLD (ANTI BIAS)
    # =========================
    if score >= 2:
        return "positif"
    elif score <= -2:
        return "negatif"
    else:
        return "netral"

# =========================
# APPLY
# =========================
df['sentimen'] = df['komentar'].apply(get_sentiment)

# =========================
# SIMPAN
# =========================
df.to_csv('data/data_labeled.csv', index=False)

print("✅ Labeling selesai!")
print(df['sentimen'].value_counts())