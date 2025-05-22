import numpy as np

from flwr.client import NumPyClient
from client_storage import ClientStorage

import utils


class QuantizedRankRedClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, storage: ClientStorage, num_clients, num_bits, p, device):
        print(f"CLIENT {partition_id} INIT")
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.storage = storage
        self.num_clients = num_clients
        self.num_bits = num_bits
        self.p = p
        self.device = device

        if self.storage.prev_quant_grad is None:
            self.initialize_gradient_storage()


    def get_parameters(self, config):
        print(f"CLIENT {self.partition_id} GET PARAM")
        return []
    

    def initialize_gradient_storage(self):
        self.storage.prev_quant_grad = []
        for param in self.net.parameters():
            if param.requires_grad:
                if param.dim() == 1:
                    self.storage.prev_quant_grad.append(np.zeros(param.size()))
                elif param.dim() == 2:
                    m, n = param.size()
                    k = int(np.ceil(min(m, n) * self.p))
                    self.storage.prev_quant_grad.append([np.zeros((m, k)), np.zeros((k, )), np.zeros((k, n))])
                else:
                    new_rank = tuple(np.ceil(np.array(param.size()) * self.p).astype(int))
                    f = []
                    f.append(np.zeros(new_rank))
                    for r, i in zip(new_rank, param.size()):
                        f.append(np.zeros((i, r)))
                    self.storage.prev_quant_grad.append(f)


    def fit(self, parameters, config, verbose=True):
        print(f"CLIENT {self.partition_id} FIT")
        utils.set_parameters(self.net, parameters)

        batch_gradients = utils.get_batch_gradients(self.net, self.device, self.trainloader, flattened=False)
        red_rank_gradients = [np.array([self.partition_id])] # send the client id so the server can know to which client the dQs to be sent correspond to
        
        for i, gradient in enumerate(batch_gradients):
            if gradient.ndim == 1:
                r, q = utils.quantize_to_int(gradient, self.storage.prev_quant_grad[i], self.num_bits)
                dQ = utils.get_quantized_innovation(q, r, self.num_bits)
                Qarray = utils.get_quantized_array(dQ, self.storage.prev_quant_grad[i])
                self.storage.prev_quant_grad[i] = Qarray
                red_rank_gradients.append(q)
                red_rank_gradients.append(np.array([r]))

                if verbose:
                    print(f"Gradient shape: {gradient.shape}")
                    print(f"Quantization error: {utils.get_error(Qarray, gradient)}")
            elif gradient.ndim == 2:
                U, s, Vt = utils.svd_decomp(gradient, self.p)

                rU, qU = utils.quantize_to_int(U.flatten(), self.storage.prev_quant_grad[i][0].flatten(), self.num_bits)
                dQU = utils.get_quantized_innovation(qU, rU, self.num_bits)
                QarrayU = utils.get_quantized_array(dQU, self.storage.prev_quant_grad[i][0].flatten())
                self.storage.prev_quant_grad[i][0] = QarrayU.reshape(U.shape)

                rs, qs = utils.quantize_to_int(s, self.storage.prev_quant_grad[i][1], self.num_bits)
                dQs = utils.get_quantized_innovation(qs, rs, self.num_bits)
                Qarrays = utils.get_quantized_array(dQs, self.storage.prev_quant_grad[i][1])
                self.storage.prev_quant_grad[i][1] = Qarrays

                rVt, qVt = utils.quantize_to_int(Vt.flatten(), self.storage.prev_quant_grad[i][2].flatten(), self.num_bits)
                dQVt = utils.get_quantized_innovation(qVt, rVt, self.num_bits)
                QarrayVt = utils.get_quantized_array(dQVt, self.storage.prev_quant_grad[i][2].flatten())
                self.storage.prev_quant_grad[i][2] = QarrayVt.reshape(Vt.shape)

                red_rank_gradients.extend([qU.reshape(U.shape), np.array([rU]), qs, np.array([rs]), qVt.reshape(Vt.shape), np.array([rVt])])

                if verbose:
                    print(f"Gradient shape: {gradient.shape}")
                    print(f"Quantization error U: {utils.get_error(QarrayU.reshape(U.shape), U)}")
                    print(f"Quantization error s: {utils.get_error(Qarrays, s)}")
                    print(f"Quantization error Vt: {utils.get_error(QarrayVt.reshape(Vt.shape), Vt)}")
                    print(f"Compression ratio: {(gradient.size * 1.0) / (U.size + s.size + Vt.size)}")
                    print(f"Compression error: {utils.get_error(gradient, utils.svd_to_matrix(U, s, Vt))}")
                    print(f"Final error: {utils.get_error(gradient, utils.svd_to_matrix(self.storage.prev_quant_grad[i][0], self.storage.prev_quant_grad[i][1], self.storage.prev_quant_grad[i][2]))}")
            else:
                if verbose:
                    print(f"Gradient shape: {gradient.shape}")

                core, factors = utils.tucker_decomp(gradient, self.p)
                prev_quant_grad = self.storage.prev_quant_grad[i]
                f = []

                rCore, qCore = utils.quantize_to_int(core.flatten(), prev_quant_grad[0].flatten(), self.num_bits)
                dQCore = utils.get_quantized_innovation(qCore, rCore, self.num_bits)
                QarrayCore = utils.get_quantized_array(dQCore, prev_quant_grad[0].flatten())
                if verbose:
                    print(f"Quantization error core: {utils.get_error(QarrayCore.reshape(core.shape), core)}")
                prev_quant_grad[0] = QarrayCore.reshape(core.shape)
                f.append(qCore.reshape(core.shape))
                f.append(np.array([rCore]))

                for j, factor in enumerate(factors):
                    rf, qf = utils.quantize_to_int(factor.flatten(), prev_quant_grad[j + 1].flatten(), self.num_bits)
                    dQf = utils.get_quantized_innovation(qf, rf, self.num_bits)
                    Qarrayf = utils.get_quantized_array(dQf, prev_quant_grad[j + 1].flatten())
                    if verbose:
                        print(f"Quantization error f{j}: {utils.get_error(Qarrayf.reshape(factor.shape), factor)}")
                    prev_quant_grad[j + 1] = Qarrayf.reshape(factor.shape)
                    f.append(qf.reshape(factor.shape))
                    f.append(np.array([rf]))

                red_rank_gradients.extend(f)

                if verbose:
                    print(f"Compression ratio: {(gradient.size * 1.0) / (core.size + sum([f.size for f in factors]))}")
                    print(f"Compression error: {utils.get_error(gradient, utils.tucker_to_tensor(core, factors))}")
                    print(f"Final error: {utils.get_error(gradient, utils.tucker_to_tensor(prev_quant_grad[0], prev_quant_grad[1:]))}")

            if verbose:
                print()

        return red_rank_gradients, len(self.trainloader), {}


    def evaluate(self, parameters, config):
        return 0.0, 0, {}