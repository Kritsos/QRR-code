from typing import Union, Dict, List, Tuple, Optional
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

import numpy as np
import torch
import torch.optim as optim

import utils


class QuantizedRankRed(Strategy):
    def __init__(self, net, num_clients, num_rounds, learning_rate, num_bits, p_array, testloader, device, results_log, fraction_fit=1.0) -> None:
        super().__init__()

        self.net = net.to(device)
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.num_bits = num_bits
        self.testloader = testloader
        self.device = device
        self.results_log = results_log
        self.fraction_fit = fraction_fit
        self.min_fit_clients = num_clients
        self.min_available_clients = num_clients

        self.current_grad = None
        self.bits_transferred = 0
        self.num_comm = 0

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)

        self.quant_grad = []
        for i in range(self.num_clients):
            temp_p = []
            p = p_array[i]
            for param in net.parameters():
                if param.requires_grad:
                    if param.dim() == 1:
                        temp_p.append(np.zeros(param.size()))
                    elif param.dim() == 2:
                        m, n = param.size()
                        k = int(np.ceil(min(m, n) * p))
                        temp_p.append([np.zeros((m, k)), np.zeros((k, )), np.zeros((k, n))])
                    else:
                        new_rank = tuple(np.ceil(np.array(param.size()) * p).astype(int))
                        f = []
                        f.append(np.zeros(new_rank))
                        for r, i in zip(new_rank, param.size()):
                            f.append(np.zeros((i, r)))
                        temp_p.append(f)

            self.quant_grad.append(temp_p)


    def __repr__(self) -> str:
        return "RankRed"


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        print("SERVER INIT")
        ndarrays = utils.get_parameters(self.net)
        return ndarrays_to_parameters(ndarrays)
    

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        print("SERVER CONFIG FIT")
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []
        for client in clients:
            fit_configurations.append((client, FitIns(parameters, {})))

        return fit_configurations


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print("SERVER AGGR FIT")
        for failure in failures:
            print(f"Failure: {failure}")

        gradients = []
        for _, fit_res in results:
            curr_grads = []
            data = parameters_to_ndarrays(fit_res.parameters)
            client_id = data[0][0]
            arrays = data[1:]

            i = 0
            quant_grad_i = 0
            while(i < len(arrays)):
                q = arrays[i]
                r = arrays[i + 1]
                if q.ndim == 1:
                    self.quant_grad[client_id][quant_grad_i] += utils.get_quantized_innovation(q, r, self.num_bits)
                    curr_grads.append(self.quant_grad[client_id][quant_grad_i])
                    self.bits_transferred += 32 + self.num_bits * q.size
                    i += 2
                elif q.ndim == 2: # q = qU
                    qs = arrays[i + 2]
                    rs = arrays[i + 3]
                    qVt = arrays[i + 4]
                    rVt = arrays[i + 5]

                    dQU = utils.get_quantized_innovation(q.flatten(), r, self.num_bits)
                    self.quant_grad[client_id][quant_grad_i][0] += dQU.reshape(q.shape)

                    dQs = utils.get_quantized_innovation(qs, rs, self.num_bits)
                    self.quant_grad[client_id][quant_grad_i][1] += dQs

                    dQVt = utils.get_quantized_innovation(qVt.flatten(), rVt, self.num_bits)
                    self.quant_grad[client_id][quant_grad_i][2] += dQVt.reshape(qVt.shape)

                    curr_grads.append(utils.svd_to_matrix(self.quant_grad[client_id][quant_grad_i][0], self.quant_grad[client_id][quant_grad_i][1], self.quant_grad[client_id][quant_grad_i][2]))
                    self.bits_transferred += 3 * 32 + self.num_bits * (q.size + qs.size + qVt.size)
                    i += 6
                else: # q = qCore
                    dQcore = utils.get_quantized_innovation(q.flatten(), r, self.num_bits)
                    self.quant_grad[client_id][quant_grad_i][0] += dQcore.reshape(q.shape)
                    self.bits_transferred += 32 + self.num_bits * q.size

                    n = q.ndim
                    qf = arrays[i + 2:i + 2 * n + 2:2]
                    rf = arrays[i + 3:i + 2 * n + 2:2]
                    for j, (qfactor, rfactor) in enumerate(zip(qf, rf)):
                        dQf = utils.get_quantized_innovation(qfactor.flatten(), rfactor, self.num_bits)
                        self.quant_grad[client_id][quant_grad_i][j + 1] += dQf.reshape(qfactor.shape)
                        self.bits_transferred += 32 + self.num_bits * qfactor.size

                    curr_grads.append(utils.tucker_to_tensor(self.quant_grad[client_id][quant_grad_i][0], self.quant_grad[client_id][quant_grad_i][1:]))
                    i += 2 * n + 2
                quant_grad_i += 1

            gradients.append(curr_grads)

        self.num_comm += len(results)

        mean_gradient = [np.sum([gradients[i][j] for i in range(len(gradients))], axis=0) for j in range(len(gradients[0]))]
        
        self.optimizer.zero_grad()
        i = 0
        for param in self.net.parameters():
            if param.requires_grad:
                curr_grad = mean_gradient[i]
                i += 1

                param.grad = torch.from_numpy(curr_grad).float().to(self.device)

        self.optimizer.step()

        loss, accuracy = utils.test(self.net, self.device, self.testloader)
        metrics_aggregated = {"loss": loss, "accuracy": accuracy, "bits transferred": self.bits_transferred}

        self.results_log.log("iteration", server_round)
        self.results_log.log("bits", self.bits_transferred)
        self.results_log.log("communications", self.num_comm)
        self.results_log.log("loss", loss)
        self.results_log.log("accuracy", accuracy)
        self.results_log.log("gradient", np.linalg.norm(np.concatenate([grad.flatten() for grad in mean_gradient])))

        return ndarrays_to_parameters(utils.get_parameters(self.net)), metrics_aggregated
    

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        print("SERVER CONFIG EVAL")
        return []
    

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        print("SERVER AGGR EVAL")
        return None, {}


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        print("SERVER EVAL")
        loss, accuracy = utils.test(self.net, self.device, self.testloader)

        if server_round == 0:
            self.results_log.log("iteration", 0)
            self.results_log.log("bits", self.bits_transferred)
            self.results_log.log("communications", 0)
            self.results_log.log("loss", loss)
            self.results_log.log("accuracy", accuracy)

        metrics = {"accuracy": accuracy, "bits transferred": self.bits_transferred}
        return loss, metrics


    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients