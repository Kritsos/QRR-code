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


class TwoWaySLAQ(Strategy):
    def __init__(self, net, num_clients, num_rounds, learning_rate, num_bits, testloader, device, results_log, fraction_fit=1.0) -> None:
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
        self.prev_quant_weights = None

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)


    def __repr__(self) -> str:
        return "SLAQ"


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        print("SERVER INIT")
        weights = []
        for param in self.net.parameters():
            if param.requires_grad:
                weights.append(param.detach().cpu().numpy())
        weights = np.concatenate([weight.flatten() for weight in weights])

        if self.prev_quant_weights is None:
            self.prev_quant_weights = np.zeros(weights.shape)

        r, q = utils.quantize_to_int(weights, self.prev_quant_weights, self.num_bits)
        dQ = utils.get_quantized_innovation(q, r, self.num_bits)
        self.prev_quant_weights = utils.get_quantized_array(dQ, self.prev_quant_weights)

        return ndarrays_to_parameters([q, np.array([r])])
    

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

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            q = arrays[0]
            r = arrays[1][0]

            if q.shape == (1,) and q[0] == -1:
                continue

            if self.current_grad is None:
                self.current_grad = np.zeros(q.shape)
            self.current_grad += utils.get_quantized_innovation(q, r, self.num_bits)
            # gradient -> server
            self.bits_transferred += 32 + self.num_bits * q.size
            self.num_comm += 1

        # weights -> clients
        self.bits_transferred += len(results) * self.num_bits * q.size
        self.num_comm += len(results)

        self.optimizer.zero_grad()
        grad = self.current_grad.copy()
        for param in self.net.parameters():
            if param.requires_grad:
                num_elems = param.numel()
                curr_grad = grad[:num_elems]
                grad = grad[num_elems:]
                curr_grad = np.reshape(curr_grad, param.size())
                param.grad = torch.from_numpy(curr_grad).float().to(self.device)

        self.optimizer.step()

        loss, accuracy = utils.test(self.net, self.device, self.testloader)
        metrics_aggregated = {"loss": loss, "accuracy": accuracy, "bits transferred": self.bits_transferred}

        self.results_log.log("iteration", server_round)
        self.results_log.log("bits", self.bits_transferred)
        self.results_log.log("communications", self.num_comm)
        self.results_log.log("loss", loss)
        self.results_log.log("accuracy", accuracy)
        self.results_log.log("gradient", np.linalg.norm(self.current_grad))

        weights = []
        for param in self.net.parameters():
            if param.requires_grad:
                weights.append(param.detach().cpu().numpy())
        weights = np.concatenate([weight.flatten() for weight in weights])

        r, q = utils.quantize_to_int(weights, self.prev_quant_weights, self.num_bits)
        dQ = utils.get_quantized_innovation(q, r, self.num_bits)
        self.prev_quant_weights = utils.get_quantized_array(dQ, self.prev_quant_weights)

        return ndarrays_to_parameters([q, np.array([r])]), metrics_aggregated
    

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