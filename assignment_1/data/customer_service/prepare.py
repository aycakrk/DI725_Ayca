import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import string

# 1) Yardımcı fonksiyonlar
import re

def extract_customer_messages(text):
    # Bu regex "customer:" ifadesini bulacak ve ardından gelenleri 
    # "agent:" veya metin sonuna kadar yakalayacak
    pattern = re.compile(r'(?i)customer:(.*?)(?=agent:|$)', re.DOTALL)
    """
    Açıklama:
    - (?i)  : case-insensitive
    - customer: ifadesini bul
    - (.*?)  : sonrasındaki her şeyi, 
    - (?=agent:|$) : bir sonraki 'agent:' veya metin sonuna ($) gelene kadar yakala
    - re.DOTALL  : . karakteri \n dahil her şeyi temsil etsin
    """
    matches = pattern.findall(text)

    # matches bir dizi string döndürecek; her string "customer:"dan 
    # "agent:"a kadarki kısım (veya satır sonuna kadar)
    # Bu kısımlar "customer:" ifadesi olmadan yakalanacak
    # isterseniz .strip() ile boşlukları temizleyebilirsiniz.
    cleaned_segments = []
    for seg in matches:
        # seg "hi tom im trying to log in..." gibi müşteri mesajını tutar
        seg = seg.strip()
        cleaned_segments.append(seg)

    # Tüm "customer" bölümlerini tek bir metinde birleştirmek isterseniz:
    return ' '.join(cleaned_segments)



def map_labels(labels):
    label_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}
    return [label_to_int[label] for label in labels]

def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]

def decode(ids, itos):
    return ''.join([itos[i] for i in ids if i in itos])

# 2) CSV dosyalarını oku
train_data_path = '/content/DI725_Ayca/assignment_1/data/customer_service/cleaned_train.csv'
test_data_path  = '/content/DI725_Ayca/assignment_1/data/customer_service/cleaned_test.csv'

train_df = pd.read_csv(train_data_path)
test_df  = pd.read_csv(test_data_path)

# 3) Sadece müşteri mesajlarını ayıkla
train_df['filtered_conversation'] = train_df['cleaned_conversation'].apply(extract_customer_messages)
test_df['filtered_conversation']  = test_df['cleaned_conversation'].apply(extract_customer_messages)

train_texts = train_df['filtered_conversation'].tolist()
test_texts  = test_df['filtered_conversation'].tolist()

# 4) Etiketleri integer'a çevir
train_labels = map_labels(train_df['customer_sentiment'].tolist())
test_labels  = map_labels(test_df['customer_sentiment'].tolist())

# 5) Karakter kümesi oluştur (vocab)
all_possible_chars = string.ascii_letters + string.punctuation + string.digits + ' \n'
chars = sorted(list(set(''.join(train_texts + test_texts) + all_possible_chars)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 6) Train ve Test verisini encode et
encoded_train_data = [encode(text, stoi) for text in train_texts]
encoded_test_data  = [encode(text, stoi) for text in test_texts]

# 7) Train'i train/val'e böl
train_data, val_data, train_labels, val_labels = train_test_split(
    encoded_train_data,
    train_labels,
    test_size=0.1,
    stratify=train_labels,
    random_state=42
)

# 8) Kayıt yeri
output_dir = '/content/DI725_Ayca/assignment_1/data/customer_service/'
os.makedirs(output_dir, exist_ok=True)

# A) Train konuşmaları "object array" haline getirip .npy kaydet
train_data_obj = np.array(train_data, dtype=object)
np.save(os.path.join(output_dir, 'train_data.npy'), train_data_obj, allow_pickle=True)

train_labels_arr = np.array(train_labels, dtype=np.uint16)  # Etiketler sabit boyutlu -> normal array
np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels_arr)

# B) Validation konuşmaları
val_data_obj = np.array(val_data, dtype=object)
np.save(os.path.join(output_dir, 'val_data.npy'), val_data_obj, allow_pickle=True)

val_labels_arr = np.array(val_labels, dtype=np.uint16)
np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels_arr)

# C) Test konuşmaları
test_data_obj = np.array(encoded_test_data, dtype=object)
np.save(os.path.join(output_dir, 'test_data.npy'), test_data_obj, allow_pickle=True)

test_labels_arr = np.array(test_labels, dtype=np.uint16)
np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels_arr)

# 9) Meta bilgileri (vocab, stoi vs.)
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# 10) Basit kontrol
sample_text = "Test encoding and decoding!"
encoded_text = encode(sample_text, stoi)
decoded_text = decode(encoded_text, itos)
print("Original:", sample_text)
print("Encoded:", encoded_text)
print("Decoded:", decoded_text)
