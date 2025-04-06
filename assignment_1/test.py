import os
import sys
import argparse
import importlib.util
import pickle
import random
import numpy as np
import torch

from model import GPT, GPTConfig

#####################################
# Veri Yükleme & Batch İşlemleri 
#####################################
def load_data(data_dir):
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True)
    val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'), allow_pickle=True)
    test_data = None
    test_labels = None
    test_path_data = os.path.join(data_dir, 'test_data.npy')
    test_path_labels = os.path.join(data_dir, 'test_labels.npy')
    if os.path.exists(test_path_data) and os.path.exists(test_path_labels):
        test_data = np.load(test_path_data, allow_pickle=True)
        test_labels = np.load(test_path_labels, allow_pickle=True)
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def get_batch_conversations(data_list, labels_list, batch_size, block_size, device):
    n = len(data_list)
    ix = random.sample(range(n), k=batch_size)
    X_list, Y_list, S_list = [], [], []
    for i in ix:
        tokens = data_list[i]
        label = labels_list[i]
        if len(tokens) > block_size:
            tokens = tokens[:block_size]
        else:
            pad_len = block_size - len(tokens)
            tokens = np.concatenate([tokens, np.zeros(pad_len, dtype=np.int64)])
        x = tokens[:-1]
        y = tokens[1:]
        X_list.append(x)
        Y_list.append(y)
        S_list.append(label)
    X_torch = torch.tensor(X_list, dtype=torch.long, device=device)
    Y_torch = torch.tensor(Y_list, dtype=torch.long, device=device)
    S_torch = torch.tensor(S_list, dtype=torch.long, device=device)
    return X_torch, Y_torch, S_torch

#####################################
# Test Değerlendirme Fonksiyonu
#####################################
def evaluate_model_on_test(model, test_data, test_labels, block_size, device):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for i in range(len(test_data)):
            tokens = test_data[i]
            label = test_labels[i]
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            else:
                pad_len = block_size - len(tokens)
                tokens = np.concatenate([tokens, np.zeros(pad_len, dtype=np.int64)])
            x_t = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            _, sentiment_logits, _ = model(x_t, sentiment_labels=None, targets=None)
            pred_sent = sentiment_logits.argmax(dim=1).item()
            preds.append(pred_sent)
            gts.append(label)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(gts, preds, labels=[0, 1, 2])
    print("Confusion Matrix (rows=TRUE, cols=PRED) [neg, neu, pos]:")
    print(cm)
    target_names = ["Negative", "Neutral", "Positive"]
    print("\nClassification Report:")
    print(classification_report(gts, preds, target_names=target_names, digits=4))
    return cm

#####################################
# Ana Test Döngüsü
#####################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Konfigürasyon dosyasının yolu")
    parser.add_argument("--checkpoint", type=str, required=True, help="Yüklemek için model checkpoint dosyası")
    args = parser.parse_args()

    # Konfigürasyon dosyasını yükle
    spec = importlib.util.spec_from_file_location("config", args.config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device =", device)

    data_dir      = config_module.data_dir
    block_size    = config_module.block_size

    # Veri yükleme (test verileri de dahil)
    (_, _), (_, _), (test_data, test_labels) = load_data(data_dir)
    if test_data is None or test_labels is None:
        print("Test verileri bulunamadı!")
        return

    # meta.pkl'den vocab_size bilgisini güncelle
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta.get('vocab_size', config_module.vocab_size)
    print("Loaded vocab_size =", vocab_size)

    # Model konfigürasyonu oluşturma
    from model import GPTConfig
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=config_module.n_layer,
        n_head=config_module.n_head,
        n_embd=config_module.n_embd,
        dropout=config_module.dropout,
        bias=config_module.bias
    )
    from model import GPT
    model = GPT(model_config)
    model.to(device)

    # Checkpoint yükleme
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint yüklendi: {args.checkpoint}")

    # Test verileri üzerinde değerlendirme
    evaluate_model_on_test(model, test_data, test_labels, block_size, device)

if __name__ == '__main__':
    main()
