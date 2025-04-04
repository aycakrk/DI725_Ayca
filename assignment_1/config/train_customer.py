# config/train_don_char.py

out_dir = '/content/DI725_Ayca/assignment_1/data/customer_service'
eval_interval = 250
eval_iters = 200
log_interval = 50
always_save_checkpoint = False

wandb_log = True
wandb_project = 'customer_service'
wandb_run_name = 'mini-gpt'

dataset = 'customer_service'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
compile = False
