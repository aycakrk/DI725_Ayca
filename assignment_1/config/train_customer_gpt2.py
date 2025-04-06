# Data paths
train_csv = "/content/DI725_Ayca/assignment_1/data/customer_service/cleaned_train.csv"
test_csv  = "/content/DI725_Ayca/assignment_1/data/customer_service/cleaned_test.csv"

# Hyperparameters & settings
max_length = 128                   # Maksimum token sayısı
num_train_epochs = 3               # Eğitim dönemi sayısı
train_batch_size = 2               # Her cihazda eğitim batch boyutu
eval_batch_size = 2                # Her cihazda doğrulama/test batch boyutu
learning_rate = 1e-5               # Öğrenme hızı
weight_decay = 0.01                # Ağırlık çürümesi
output_dir = "gpt2_finetune_output"  # Çıktı dizini
save_steps = 500                   # Kaç adımda model checkpoint kaydedilecek
logging_steps = 100                # Kaç adımda log yazılacak

# Model settings
model_name = "gpt2"                # Kullanılacak önceden eğitilmiş model
