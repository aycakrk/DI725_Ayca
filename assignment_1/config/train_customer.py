# Eğitim ve model parametreleri

# Veri ve model parametreleri
data_dir = '/content/DI725_Ayca/assignment_1/data/customer_service'
block_size = 128
batch_size = 16
max_iters = 2000
eval_interval = 200
eval_iters = 50
learning_rate = 1e-3
weight_decay = 1e-2

# Modelin yapılandırması için
vocab_size = 2000  # Bu değer meta.pkl'den güncellenecektir
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1
bias = True

# Ekstra ayarlar
compile_model = False  # Komut satırından --compile ile override edilebilir
