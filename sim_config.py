import torch
from client_storage import ClientStorage
from results_log import ResultsLog
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


######## CHANGE ONLY THESE PARAMETERS ########

# number of workers
NUM_CLIENTS = 10
# number of federated learning iterations
NUM_ROUNDS = 1
# how many bits to use when quantizing the gradients
NUM_BITS = 8
# number of weights differences to consider for the threshold calculation of eq(9)
D = 10
# coefficient of each gradient approximation (differences)
XI = np.array([1.0 / D for _ in range(D)])
# maximum number of consecutive rounds that a client can skip uploading the gradient
TIMER_MAX = 50
# rank reduction percentage (should be an iterable, with a value for each client)
P1 = [0.1] * NUM_CLIENTS
P2 = [0.2] * NUM_CLIENTS
P3 = [0.3] * NUM_CLIENTS
P_MIXED = np.linspace(0.1, 0.3, NUM_CLIENTS)
# learning rate
LEARNING_RATE = 0.001
# batch size
BATCH_SIZE = 512
# device on which to train
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# UNCOMMENT THE DATASET YOU WISH TO USE

# # MNIST
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_size = len(mnist_trainset)
# subset_size = train_size // NUM_CLIENTS
# lengths = [subset_size] * (NUM_CLIENTS - 1) + [train_size - subset_size * (NUM_CLIENTS - 1)]
# subsets = random_split(mnist_trainset, lengths)
# TRAINLOADERS = [DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True) for subset in subsets]
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# TESTLOADER = DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=False)

# # Fashion MNIST
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# fmnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# train_size = len(fmnist_trainset)
# subset_size = train_size // NUM_CLIENTS
# lengths = [subset_size] * (NUM_CLIENTS - 1) + [train_size - subset_size * (NUM_CLIENTS - 1)]
# subsets = random_split(fmnist_trainset, lengths)
# TRAINLOADERS = [DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True) for subset in subsets]
# fmnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# TESTLOADER = DataLoader(fmnist_testset, batch_size=BATCH_SIZE, shuffle=False)

# CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = len(cifar10_trainset)
subset_size = train_size // NUM_CLIENTS
lengths = [subset_size] * (NUM_CLIENTS - 1) + [train_size - subset_size * (NUM_CLIENTS - 1)]
subsets = random_split(cifar10_trainset, lengths)
TRAINLOADERS = [DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True) for subset in subsets]
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
TESTLOADER = DataLoader(cifar10_testset, batch_size=BATCH_SIZE, shuffle=False)

##############################################

######## DO NOT CHANGE ########

# Defining these variables here ensures that when the main run_simulation.py file spawns client actors using Ray
# these variables will live inside that actor and the variables state will be updated across training iterations,
# instead of being reinitialized at each iteration
SLAQ_CLIENT_STORAGE = [ClientStorage(D) for _ in range(NUM_CLIENTS)]
TWO_WAY_SLAQ_CLIENT_STORAGE = [ClientStorage(D) for _ in range(NUM_CLIENTS)]
QRR1_STORAGE = [ClientStorage(D) for _ in range(NUM_CLIENTS)]
QRR2_STORAGE = [ClientStorage(D) for _ in range(NUM_CLIENTS)]
QRR3_STORAGE = [ClientStorage(D) for _ in range(NUM_CLIENTS)]


# The issue with the client storage objects is not encountered with these variables since the server runs in the main
# python process and not in a separate Ray actor, so they could as well be declared in run_simulation.py
SGD_RESULTS_LOG = ResultsLog()
SLAQ_RESULTS_LOG = ResultsLog()
TWO_WAY_SLAQ_RESULTS_LOG = ResultsLog()
QRR1_RESULTS_LOG = ResultsLog()
QRR2_RESULTS_LOG = ResultsLog()
QRR3_RESULTS_LOG = ResultsLog()