import numpy as np
import torch

from client_storage import ClientStorage
from flwr.client import NumPyClient

import utils


class TwoWaySLAQClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, storage: ClientStorage, num_clients, num_bits, xi, a, device, timer_max):
        print(f"CLIENT {partition_id} INIT")
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.storage = storage
        self.num_clients = num_clients
        self.num_bits = num_bits
        self.xi = xi
        self.a = a
        self.device = device
        self.timer_max = timer_max


    def get_parameters(self, config):
        print(f"CLIENT {self.partition_id} GET PARAM")
        return []


    def get_gradient_innovation(self, gradient):
        first_round = self.storage.prev_quant_grad is None
        if first_round:
            self.storage.prev_quant_grad = np.zeros(gradient.shape)

        r, q = utils.quantize_to_int(gradient, self.storage.prev_quant_grad, self.num_bits)
        dQ = utils.get_quantized_innovation(q, r, self.num_bits)
        Qarray = utils.get_quantized_array(dQ, self.storage.prev_quant_grad)
        err = utils.get_error(gradient, Qarray)

        if not first_round and self.storage.timer < self.timer_max and utils.should_skip_upload(Qarray, self.storage.prev_quant_grad, err, self.storage.prev_quant_error,
                                                                            self.a, self.num_clients, self.xi, self.storage.prev_weights, verbose=True):
            self.storage.timer += 1
            return [np.array([-1]), np.array([-1])]
            
        self.storage.timer = 0
        self.storage.prev_quant_grad = Qarray
        self.storage.prev_quant_error = err

        return [q, np.array([r])]


    def fit(self, parameters, config):
        print(f"CLIENT {self.partition_id} FIT")
        q = parameters[0]
        r = parameters[1][0]
        dQ = utils.get_quantized_innovation(q, r, self.num_bits)
        if self.storage.prev_quant_weights is None:
            self.storage.prev_quant_weights = np.zeros(q.shape)
        quant_weights = utils.get_quantized_array(dQ, self.storage.prev_quant_weights)
        self.storage.prev_quant_weights = quant_weights
        self.storage.prev_weights.append(quant_weights)

        curr_quant_weights = quant_weights.copy()
        with torch.no_grad():
            for param in self.net.parameters():
                if param.requires_grad:
                    num_elems = param.numel()
                    curr_weights = curr_quant_weights[:num_elems]
                    curr_quant_weights = curr_quant_weights[num_elems:]
                    curr_weights = np.reshape(curr_weights, param.size())
                    param.copy_(torch.from_numpy(curr_weights).float().to(self.device))

        gradient_mean = utils.get_batch_gradients(self.net, self.device, self.trainloader)
        return self.get_gradient_innovation(gradient_mean), len(self.trainloader), {}


    def evaluate(self, parameters, config):
        return 0.0, 0, {}