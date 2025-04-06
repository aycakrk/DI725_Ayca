import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report
import importlib.util

# Dinamik olarak config dosyasını yükle
spec = importlib.util.spec_from_file_location("config", "/content/DI725_Ayca/assignment_1/config/train_customer_gpt2.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

#############################################
# 1) Data Preparation
#############################################
train_df = pd.read_csv(config.train_csv)
test_df  = pd.read_csv(config.test_csv)

# CSV’de bulunan "customer_sentiment" sütununa göre etiketleri belirleyelim
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df["label"] = train_df["customer_sentiment"].map(label_mapping)
test_df["label"]  = test_df["customer_sentiment"].map(label_mapping)

# Eğitim verisini stratified olarak train ve validation setlerine ayırıyoruz
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

#############################################
# 2) Tokenizer & Model Setup (GPT‑2)
#############################################
model_name = config.model_name
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# GPT‑2’nin pad token’ı yoktur, o yüzden eos token’ı pad olarak kullanıyoruz
tokenizer.pad_token = tokenizer.eos_token

# GPT‑2 modelini, 3 etiketli sequence classification için yükleyelim
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.config.pad_token_id = tokenizer.eos_token_id

#############################################
# 3) Tokenization
#############################################
def tokenize_texts(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)

train_encodings = tokenize_texts(train_df["cleaned_conversation"].tolist())
val_encodings   = tokenize_texts(val_df["cleaned_conversation"].tolist())
# Test verisini eğitimde kullanmayacağız; test aşamasında tokenize edilecektir.

#############################################
# 4) Create PyTorch Datasets
#############################################
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_df["label"].tolist())
val_dataset   = SentimentDataset(val_encodings, val_df["label"].tolist())

#############################################
# 5) Weighted Random Sampler
#############################################
labels_array = np.array(train_df["label"].tolist())
class_counts = np.bincount(labels_array)
print("Class counts:", class_counts)

# Her örnek için ağırlık; azınlık sınıflara daha yüksek ağırlık veriliyor
weights = 1.0 / class_counts[labels_array]
weights = torch.DoubleTensor(weights)

sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def custom_get_train_dataloader():
    return DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, sampler=sampler)

#############################################
# 6) Compute Metrics Function
#############################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

#############################################
# 7) Training Arguments & Trainer
#############################################
training_args = TrainingArguments(
    output_dir=config.output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    save_steps=config.save_steps,
    logging_steps=config.logging_steps,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Özel sampler’ımızı kullanmak için get_train_dataloader fonksiyonunu override ediyoruz
trainer.get_train_dataloader = custom_get_train_dataloader

#############################################
# 8) Fine-Tuning & Checkpoint Save
#############################################
trainer.train()
trainer.save_model("gpt2_finetune_checkpoint")
print("Eğitim tamamlandı, model 'gpt2_finetune_checkpoint' dizinine kaydedildi.")
