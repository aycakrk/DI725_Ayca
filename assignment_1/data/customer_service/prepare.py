import os
import pickle
import numpy as np
import pandas as pd

# Dosya yolları
train_csv_path = '/content/DI725_Ayca/assignment_1/data/customer_service/train.csv'
test_csv_path = '/content/DI725_Ayca/assignment_1/data/customer_service/test.csv'
out_dir = '/content/DI725_Ayca/assignment_1/data/customer_service'

# Dosyaların olup olmadığını kontrol et
if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"{train_csv_path} bulunamadı.")
if not os.path.exists(test_csv_path):
    raise FileNotFoundError(f"{test_csv_path} bulunamadı.")

# Eğitim verisini oku
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Tüm metinleri birleştir (karakter tabanlı işlem yapmak için)
train_data = ''.join(train_df['conversation'].astype(str).tolist())
test_data = ''.join(test_df['conversation'].astype(str).tolist())

# Tüm metinleri birleştirip benzersiz karakterleri bul
data = train_data + test_data
print(f"length of dataset in characters: {len(data):,}")

# Unique karakterleri bul ve eşleştir
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size}")

# Karakterleri integerlara eşle
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode ve Decode fonksiyonları
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Eğitim ve doğrulama kümelerini ayır
n = len(train_data)
train_ids = encode(train_data[:int(0.9 * n)])
val_ids = encode(train_data[int(0.9 * n):])

# Verilerin boyutlarını yazdır
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Encode edilmiş verileri numpy array olarak kaydet
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Çıktı dizinini oluştur
os.makedirs(out_dir, exist_ok=True)

# Dosyalara kaydet
train_ids.tofile(os.path.join(out_dir, 'train.bin'))
val_ids.tofile(os.path.join(out_dir, 'val.bin'))

# Meta dosyasını kaydet
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("✅ Veri başarıyla işlendi ve kaydedildi.")
