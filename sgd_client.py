from flwr.client import NumPyClient

import utils


class SGDClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, device):
        print(f"CLIENT {partition_id} INIT")
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.device = device


    def get_parameters(self, config):
        print(f"CLIENT {self.partition_id} GET PARAM")
        return []


    def fit(self, parameters, config):
        print(f"CLIENT {self.partition_id} FIT")
        utils.set_parameters(self.net, parameters)

        gradient_mean = utils.get_batch_gradients(self.net, self.device, self.trainloader)
        return [gradient_mean], len(self.trainloader), {}


    def evaluate(self, parameters, config):
        return 0.0, 0, {}