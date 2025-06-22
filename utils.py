from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
import tensorly
import numpy as np

from collections import OrderedDict
from typing import List, Tuple

from sim_config import DEVICE


tensorly.set_backend('pytorch')

######## CHOOSE NETWORK ARCHITECTURE ########
# The class should always be named Net

# ## Convolutional Network
# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()

#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.fc1 = nn.Linear(32 * 14 * 14, 64)
#         self.relu3 = nn.ReLU()

#         self.fc2 = nn.Linear(64, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.relu1(x)

#         x = self.conv2(x)
#         x = self.relu2(x)

#         x = self.pool(x)

#         x = x.view(x.size(0), -1) 

#         x = self.fc1(x)
#         x = self.relu3(x)

#         x = self.fc2(x)
#         return x

# ### MLP
# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(784, 200)
#         self.relu = nn.ReLU()
#         self.output = nn.Linear(200, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.hidden(x)
#         x = self.relu(x)
#         x = self.output(x)
#         return x


### VGG-LIKE
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)

        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout_fc = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
    
#############################################
    

def set_parameters(net, parameters):
    state_dict = {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)}
    for key, param in state_dict.items():
        if param.numel() == 0:
            print(f"Warning: Parameter {key} is empty!")
            state_dict[key] = torch.zeros_like(net.state_dict()[key])
    net.load_state_dict(state_dict, strict=False)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
    

def get_batch_gradients(net: nn.Module, device: str, trainloader: DataLoader, flattened=True, evaluate=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    x, labels = next(iter(trainloader))
    ### uncomment if neural network does not contain convolutional layers at the start (for MNIST)
    # x = x.view(-1, 784).to(device)
    x = x.to(device)
    labels = labels.to(device)

    output = net(x)
    loss = criterion(output, labels)

    net.zero_grad()
    loss.backward()

    batch_gradients = []
    for param in net.parameters():
        if param.requires_grad:
            batch_gradients.append(param.grad.cpu().numpy())

    if flattened:
        gradient_mean = np.concatenate([gradient.flatten() for gradient in batch_gradients])
    else:
        gradient_mean = batch_gradients

    if evaluate:
        correct, total_samples, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in trainloader:
                x, labels = batch
                x = x.view(-1, 784)
                x = x.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)
                    
                outputs = net(x)
                loss = criterion(outputs, labels)
                    
                running_loss += loss.item()
                total_samples += batch_size
                correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            
        running_loss /= len(trainloader)
        acc = correct / total_samples
        print(f"Train loss: {running_loss}, Accuracy: {acc}")
    
    return gradient_mean


def test(net: nn.Module, device: str, testloader: DataLoader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            x, labels = batch
            ### uncomment if neural network does not contain convolutional layers at the start (for MNIST)
            # x = x.view(-1, 784).to(device)
            x = x.to(device)
            labels = labels.to(device)
            outputs = net(x)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy


def quantize_to_int(array: np.ndarray, prev_quantized_array: np.ndarray, num_bits: int) -> Tuple[float, np.ndarray]:
    """ Calculates eq (5) """
    radius = np.max(np.abs(array - prev_quantized_array))

    if radius == 0: # only encountered it with 1x1 arrays (a single number), in some of the factor matrices of the tucker decomposition (when input channels of a conv layer is 1)
        return radius, np.zeros(array.shape)

    t = 1.0 / (2 ** num_bits - 1)
    q = np.floor((array - prev_quantized_array + radius) / (2 * t * radius) + 0.5)

    return radius, q


def get_quantized_innovation(q: np.ndarray, radius: float, num_bits: int) -> np.ndarray:
    """ Calculates eq (6) """
    t = 1.0 / (2 ** num_bits - 1)
    return 2 * t * radius * q - radius


def get_quantized_array(delta_q: np.ndarray, prev_quantized_array: np.ndarray) -> np.ndarray:
    """ Calculates eq (7) """
    return prev_quantized_array + delta_q


def get_error(array: np.ndarray, quantized_array: np.ndarray) -> float:
    e = array - quantized_array
    return np.power(np.linalg.norm(e), 2)


def should_skip_upload(current_quantized_array: np.ndarray, prev_quantized_array: np.ndarray, curr_err: float, prev_err: float, a: float, M: int, xi: np.ndarray, weights: List[np.ndarray], verbose=False) -> bool:
    """ Evaluates eq (9a) """
    dQ = np.power(np.linalg.norm(prev_quantized_array - current_quantized_array), 2)

    D = xi.shape[0]
    s = 0
    for i in range(D):
        if weights[D - 1 - i] is None:
            continue
        s = s + xi[i] * np.power(np.linalg.norm(weights[D - i] - weights[D - 1 - i]), 2)
    threshold = (1 / (a * a * M * M)) * s + 3 * (curr_err + prev_err)
    skip = dQ <= threshold

    if verbose:
        print(f"{(1 / (a * a * M * M))} * {s} + 3 * ({curr_err} + {prev_err})")
        if skip:
            print(f"skip: {dQ} <= {threshold}")
        else:
            print(f"upload: {dQ} > {threshold}")

    return skip


def svd_decomp(array: np.ndarray, p: float) -> List[np.ndarray]:
    k = int(np.ceil(min(array.shape) * p))
    array = torch.from_numpy(array).to(DEVICE)
    U, S, Vt = torch.linalg.svd(array)
    return U[:, :k].cpu().numpy(), S[:k].cpu().numpy(), Vt[:k, :].cpu().numpy()


def tucker_decomp(tensor: np.ndarray, p: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    new_rank = tuple(np.ceil(np.array(tensor.shape) * p).astype(int))
    tensor = torch.from_numpy(tensor).to(DEVICE)
    core, factors = tensorly.decomposition.tucker(tensor, rank=new_rank)
    core = core.cpu().numpy()
    factors = [factor.cpu().numpy() for factor in factors]
    return core, factors


def svd_to_matrix(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    U, S, Vt = torch.from_numpy(U).to(DEVICE), torch.from_numpy(S).to(DEVICE), torch.from_numpy(Vt).to(DEVICE)
    S = torch.diag(S)
    return (U @ S @ Vt).cpu().numpy()


def tucker_to_tensor(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    core = torch.from_numpy(core).to(DEVICE)
    factors = [torch.from_numpy(factor).to(DEVICE) for factor in factors]
    return (tensorly.tucker_to_tensor((core, factors))).cpu().numpy()