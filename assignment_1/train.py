import os
import sys
import argparse
import importlib.util
import pickle
import random
import numpy as np
import torch
import wandb  # wandb importu

from model_weighted_loss import GPT, GPTConfig

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
# Eğitim & Doğrulama Fonksiyonları
#####################################
def estimate_loss(model, data_list, labels_list, batch_size, block_size, device, eval_iters=50):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y, S = get_batch_conversations(data_list, labels_list, batch_size, block_size, device)
        with torch.no_grad():
            _, _, loss = model(X, sentiment_labels=S, targets=Y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

#####################################
# Ana Eğitim Döngüsü
#####################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Konfigürasyon dosyasının yolu")
    parser.add_argument("--compile", type=str, default=None, help="Model derleme opsiyonu (True/False)")
    parser.add_argument("--save_path", type=str, default="checkpoint.pt", help="Kaydedilecek model dosyası")
    args = parser.parse_args()

    # Konfigürasyon dosyasını dinamik olarak yükle
    spec = importlib.util.spec_from_file_location("config", args.config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if args.compile is not None:
        config_module.compile_model = args.compile.lower() == 'true'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device =", device)

    # Yalnızca gerekli konfigürasyon parametrelerini al
    config_dict = {
        'data_dir': config_module.data_dir,
        'block_size': config_module.block_size,
        'batch_size': config_module.batch_size,
        'max_iters': config_module.max_iters,
        'eval_interval': config_module.eval_interval,
        'eval_iters': config_module.eval_iters,
        'learning_rate': config_module.learning_rate,
        'weight_decay': config_module.weight_decay,
        'vocab_size': config_module.vocab_size,
        'n_layer': config_module.n_layer,
        'n_head': config_module.n_head,
        'n_embd': config_module.n_embd,
        'dropout': config_module.dropout,
        'bias': config_module.bias,
        'compile_model': config_module.compile_model
    }

    # W&B başlatma
    wandb.init(
        project=getattr(config_module, "wandb_project", "default_project"),
        config=config_dict
    )

    # Konfigürasyon parametreleri
    data_dir      = config_module.data_dir
    block_size    = config_module.block_size
    batch_size    = config_module.batch_size
    max_iters     = config_module.max_iters
    eval_interval = config_module.eval_interval
    eval_iters    = config_module.eval_iters
    learning_rate = config_module.learning_rate
    weight_decay  = config_module.weight_decay

    # Veri yükleme
    (train_data, train_labels), (val_data, val_labels), _ = load_data(data_dir)

    # meta.pkl'den vocab_size bilgisini güncelle
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta.get('vocab_size', config_module.vocab_size)
    print("Loaded vocab_size =", vocab_size)

    # Model konfigürasyonu ve oluşturma
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=config_module.n_layer,
        n_head=config_module.n_head,
        n_embd=config_module.n_embd,
        dropout=config_module.dropout,
        bias=config_module.bias
    )
    model = GPT(model_config)
    model.to(device)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (0.9, 0.99), device)

    # Eğitim döngüsü
    for iter_num in range(max_iters):
        X, Y, S = get_batch_conversations(train_data, train_labels, batch_size, block_size, device)
        _, _, loss = model(X, sentiment_labels=S, targets=Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrikleri wandb'ye loglama
        wandb.log({"train_loss": loss.item(), "iteration": iter_num})

        if iter_num % 50 == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}")

        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = estimate_loss(model, val_data, val_labels, batch_size, block_size, device, eval_iters=eval_iters)
            print(f"Step {iter_num}: val_loss = {val_loss:.4f}")
            wandb.log({"val_loss": val_loss, "iteration": iter_num})

    # Eğitim sonunda checkpoint kaydet
    torch.save(model.state_dict(), args.save_path)
    print(f"Model kaydedildi: {args.save_path}")
    wandb.save(args.save_path)
    wandb.finish()

if __name__ == '__main__':
    main()
