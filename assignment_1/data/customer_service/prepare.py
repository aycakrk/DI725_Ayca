import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Step 1: Define a function to extract only customer messages
def extract_customer_messages(text):
    lines = text.split('\n')
    customer_lines = [line for line in lines if line.lower().startswith('customer:')]
    return ' '.join(customer_lines).replace('customer: ', '')

# Step 2: Define a function to map sentiment labels to integers
def map_labels(labels):
    label_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}
    return [label_to_int[label] for label in labels]

# Step 3: Define a function to encode text using character-to-integer mapping
def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]

# Step 4: Load training dataset
train_data_path = '/content/DI725_Ayca/assignment_1/data/customer_service/cleaned_train.csv'
train_df = pd.read_csv(train_data_path)

# Apply filtering step (Extract only customer messages)
train_df['filtered_conversation'] = train_df['cleaned_conversation'].apply(extract_customer_messages)

train_data = train_df['filtered_conversation'].tolist()
train_labels = map_labels(train_df['customer_sentiment'].tolist())

# Step 5: Load test dataset
test_data_path = '/content/DI725_Ayca/assignment_1/data/customer_service/cleaned_test.csv'
test_df = pd.read_csv(test_data_path)

# Apply filtering step
test_df['filtered_conversation'] = test_df['cleaned_conversation'].apply(extract_customer_messages)

test_data = test_df['filtered_conversation'].tolist()
test_labels = map_labels(test_df['customer_sentiment'].tolist())

# Step 6: Create a mapping from characters to integers and vice versa
import string

# New way to generate the character set
# Adding all common characters including uppercase, lowercase, punctuation, and digits
all_possible_chars = string.ascii_letters + string.punctuation + string.digits + ' \n'  # Including space and newline
chars = sorted(list(set(''.join(train_data + test_data) + all_possible_chars)))  # Combine with your data

# Create your mappings as usual
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# Step 7: Encode the data
encoded_train_data = [encode(text, stoi) for text in train_data]
encoded_test_data = [encode(text, stoi) for text in test_data]

# Step 8: Split training data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    encoded_train_data, train_labels, test_size=0.1, stratify=train_labels, random_state=42)

# Step 9: Save data without padding (Each conversation is saved individually)
output_dir = '/content/DI725_Ayca/assignment_1/data/customer_service/'

# Save training data properly (each conversation separately)
with open(os.path.join(output_dir, 'train_data.bin'), 'wb') as f:
    for conversation in train_data:
        length = len(conversation)
        np.array([length], dtype=np.uint16).tofile(f)  # Save length of conversation first
        np.array(conversation, dtype=np.uint16).tofile(f)

# Save validation data properly
with open(os.path.join(output_dir, 'val_data.bin'), 'wb') as f:
    for conversation in val_data:
        length = len(conversation)
        np.array([length], dtype=np.uint16).tofile(f)
        np.array(conversation, dtype=np.uint16).tofile(f)

# Save test data properly
with open(os.path.join(output_dir, 'test_data.bin'), 'wb') as f:
    for conversation in encoded_test_data:
        length = len(conversation)
        np.array([length], dtype=np.uint16).tofile(f)
        np.array(conversation, dtype=np.uint16).tofile(f)


# Save labels as binary files
np.array(train_labels, dtype=np.uint16).tofile(os.path.join(output_dir, 'train_labels.bin'))
np.array(val_labels, dtype=np.uint16).tofile(os.path.join(output_dir, 'val_labels.bin'))
np.array(test_labels, dtype=np.uint16).tofile(os.path.join(output_dir, 'test_labels.bin'))

# Step 10: Save the meta information (character mapping)
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Optional: Check encoding and decoding process
def decode(ids, itos):
    return ''.join([itos[i] for i in ids if i in itos])

sample_text = "Test encoding and decoding!"
encoded_text = encode(sample_text, stoi)
decoded_text = decode(encoded_text, itos)
print("Original:", sample_text)
print("Encoded:", encoded_text)
print("Decoded:", decoded_text)
