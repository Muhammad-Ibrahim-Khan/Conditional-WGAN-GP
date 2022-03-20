import torch.cuda

#   ------- Hyperparameters -------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
NOISE_CHANNELS = 100
NUM_EPOCHS = 5
FEATURES_CRIT = 64
FEATURES_GEN = 64
CRTIC_ITERATIONS = 5
LAMBDA_GP = 10

# -------- Controller Variables --------

TRAIN = True
DATASET = 'MNIST'

# Ensure that Data location is an image folder
DATA_LOCATION = None
