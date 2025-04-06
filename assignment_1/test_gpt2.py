import pandas as pd
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report
import importlib.util

# Dinamik olarak config dosyasını yükle
spec = importlib.util.spec_from_file_location("config", "/content/DI725_Ayca/assignment_1/config/train_customer_gpt2.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

#############################################
# 1) Data Preparation (Test Verisi)
#############################################
test_df = pd.read_csv(config.test_csv)
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
test_df["label"]  = test_df["customer_sentiment"].map(label_mapping)

#############################################
# 2) Tokenizer & Model Setup (Checkpoint'ten)
#############################################
model_name = config.model_name
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Eğitim sonunda kaydedilen checkpoint’i yükleyelim
model = GPT2ForSequenceClassification.from_pretrained("gpt2_finetune_checkpoint", num_labels=3)
model.config.pad_token_id = tokenizer.eos_token_id

#############################################
# 3) Tokenization
#############################################
def tokenize_texts(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=config.max_length)

test_encodings = tokenize_texts(test_df["cleaned_conversation"].tolist())

#############################################
# 4) Create Test Dataset
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

test_dataset = SentimentDataset(test_encodings, test_df["label"].tolist())

#############################################
# 5) Evaluation using Trainer
#############################################
training_args = TrainingArguments(
    output_dir=config.output_dir,
    per_device_eval_batch_size=config.eval_batch_size,
)

trainer = Trainer(
    model=model,
    args=training_args,
)

eval_results = trainer.evaluate(test_dataset)
print("Evaluation Results:", eval_results)

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids
preds = np.argmax(logits, axis=-1)

cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
print("Confusion Matrix (rows=TRUE, cols=PRED):")
print(cm)

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Negative", "Neutral", "Positive"], digits=4))
