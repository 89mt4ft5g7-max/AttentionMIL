# config.py

# ==========================
# Model
# ==========================
INPUT_DIM = 15
HIDDEN_DIM = 64

# ==========================
# Training
# ==========================
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 1

# ==========================
# Experiment
# ==========================
SEEDS = list(range(10))      # 0～9
TEST_BAGS = range(1, 10)

# ==========================
# Optimizer
# ==========================
OPTIMIZER = "Adam"

# ==========================
# Loss
# ==========================
LOSS = "MSE"

# config.py

HIDDEN_LIST = [ 18, 20 ]
ATTENTION_LIST = [ 8 ]
LR_LIST = [ 3e-3, 4e-3, 5e-3]
EPOCH_LIST = [ 125, 150]
WEIGHT_DECAY_LIST = [ 0, 1e-5, 1e-4, 1e-3]
DROPOUT_LIST = [ 0.0, 0.1 ]
INPUT_DIM = 15
SEEDS = list(range(20))