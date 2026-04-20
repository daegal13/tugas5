import os
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

# =========================
# 📁 Pastikan folder data ada
# =========================
os.makedirs('data', exist_ok=True)

downloader = YoutubeCommentDownloader()

# =========================
# 🎯 LIST VIDEO (TAMBAHIN SEBANYAK MUNGKIN)
# =========================
urls = [
    "https://youtu.be/1aFFUbJ7ddQ?si=AOPhO13IQzsIsoIu",
    "https://youtu.be/T-H4xj5nqLY?si=WUrl6Ob6DJIKVfS7",
    "https://youtu.be/SHEeSEnSsr8?si=nGVRK4nJ2NjFE2UD",
    "https://youtu.be/T-H4xj5nqLY?si=GO7qw4FeEK75sbH0",
    "https://youtu.be/uY51ydWWDxw?si=lJDux3ftX-nWoJR4"
]

all_comments = []

# =========================
# 🔄 LOOP AMBIL KOMENTAR
# =========================
for url in urls:
    print(f"🔄 Ambil dari: {url}")
    
    try:
        count = 0
        
        for comment in downloader.get_comments_from_url(url):
            text = comment.get('text', '')
            time = comment.get('time', '')

            # filter komentar kosong
            if text.strip() != "":
                all_comments.append([
                    text,
                    time,
                    'youtube'
                ])
                count += 1

            # 🔥 BATAS PER VIDEO (biar gak lama)
            if count >= 1000:
                break

        print(f"✅ {count} komentar berhasil diambil")

    except Exception as e:
        print(f"❌ Error di {url}: {e}")

# =========================
# 💾 SIMPAN KE CSV
# =========================
df = pd.DataFrame(all_comments, columns=['komentar', 'timestamp', 'platform'])

# hapus duplikat
df.drop_duplicates(subset='komentar', inplace=True)

df.to_csv('data/data_raw.csv', index=False)

print("====================================")
print(f"🎉 TOTAL KOMENTAR: {len(df)}")
print("📁 Disimpan di: data/data_raw.csv")