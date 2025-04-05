# Output directory for saving models
out_dir = 'out-customer'
eval_interval = 250  # evaluate every 250 iterations
eval_iters = 200
log_interval = 50  # print training stats every 50 iterations

# Saving checkpoints only when validation improves
always_save_checkpoint = False

# WandB logging (disable if you are not using Weights & Biases)
wandb_log = True
wandb_project = 'customer'
wandb_run_name = 'mini-gpt'

# Dataset and data-related settings
dataset = 'customer_service'  # your dataset name
gradient_accumulation_steps = 1  # Accumulate gradients over multiple batches
batch_size = 64  # Number of samples per batch
block_size = 256  # Maximum context size for the model (length of the conversation)

# Model configuration
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # High dropout to prevent overfitting since it's a small model

# Training parameters
learning_rate = 1e-3  # Higher learning rate since it's a small model
max_iters = 2000  # Number of training iterations
lr_decay_iters = 2000  # Make equal to max_iters for a linear decay schedule
min_lr = 1e-4  # Minimum learning rate
beta2 = 0.99  # Momentum term for AdamW optimizer

warmup_iters = 100  # Number of warmup iterations to gradually increase learning rate

# Device settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False  # Do not torch.compile() the model if you face issues

# New addition for sentiment classification
num_sentiment_classes = 3  # You have 3 classes: Positive, Negative, Neutral

# This will be detected by your model
vocab_size = 96  # This is the number of unique characters in your dataset
