import copy
from flwr.common import Context
from flwr.client import Client, ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from results_log import plot_results

import sgd
import sgd_client
import slaq
import slaq_client
import two_way_slaq
import two_way_slaq_client
import quantized_rank_red
import quantized_rank_red_client
import sim_config
import utils

NET = utils.Net()


### SGD ###
def sgd_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return sgd_client.SGDClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.DEVICE).to_client()


def sgd_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=sgd.SGD(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.LEARNING_RATE, sim_config.TESTLOADER, sim_config.DEVICE, sim_config.SGD_RESULTS_LOG)
    )


### SLAQ ###
def slaq_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return slaq_client.SLAQClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.SLAQ_CLIENT_STORAGE[partition_id], sim_config.NUM_CLIENTS, 
                               sim_config.NUM_BITS, sim_config.XI, sim_config.LEARNING_RATE, sim_config.DEVICE, sim_config.TIMER_MAX).to_client()


def slaq_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=slaq.SLAQ(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.NUM_ROUNDS, sim_config.LEARNING_RATE, sim_config.NUM_BITS, sim_config.TESTLOADER, sim_config.DEVICE, sim_config.SLAQ_RESULTS_LOG)
    )


#### TWO-WAY SLAQ ###
def two_way_slaq_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return two_way_slaq_client.TwoWaySLAQClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.TWO_WAY_SLAQ_CLIENT_STORAGE[partition_id], sim_config.NUM_CLIENTS, 
                                                sim_config.NUM_BITS, sim_config.XI, sim_config.LEARNING_RATE, sim_config.DEVICE, sim_config.TIMER_MAX).to_client()


def two_way_slaq_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=two_way_slaq.TwoWaySLAQ(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.NUM_ROUNDS, sim_config.LEARNING_RATE, sim_config.NUM_BITS, sim_config.TESTLOADER, sim_config.DEVICE,
                                         sim_config.TWO_WAY_SLAQ_RESULTS_LOG)
    )


### QRR ###
def quantized_rank_red1_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return quantized_rank_red_client.QuantizedRankRedClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.QRR1_STORAGE[partition_id], sim_config.NUM_CLIENTS,
                                         sim_config.NUM_BITS, sim_config.P1[partition_id], sim_config.DEVICE).to_client()

def quantized_rank_red1_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=quantized_rank_red.QuantizedRankRed(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.NUM_ROUNDS, sim_config.LEARNING_RATE, sim_config.NUM_BITS, sim_config.P1,
                                  sim_config.TESTLOADER, sim_config.DEVICE, sim_config.QRR1_RESULTS_LOG)
    )



def quantized_rank_red2_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return quantized_rank_red_client.QuantizedRankRedClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.QRR2_STORAGE[partition_id], sim_config.NUM_CLIENTS,
                                         sim_config.NUM_BITS, sim_config.P2[partition_id], sim_config.DEVICE).to_client()

def quantized_rank_red2_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=quantized_rank_red.QuantizedRankRed(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.NUM_ROUNDS, sim_config.LEARNING_RATE, sim_config.NUM_BITS, sim_config.P2,
                                  sim_config.TESTLOADER, sim_config.DEVICE, sim_config.QRR2_RESULTS_LOG)
    )



def quantized_rank_red3_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return quantized_rank_red_client.QuantizedRankRedClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.QRR3_STORAGE[partition_id], sim_config.NUM_CLIENTS,
                                         sim_config.NUM_BITS, sim_config.P3[partition_id], sim_config.DEVICE).to_client()


def quantized_rank_red3_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=quantized_rank_red.QuantizedRankRed(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.NUM_ROUNDS, sim_config.LEARNING_RATE, sim_config.NUM_BITS, sim_config.P3,
                                  sim_config.TESTLOADER, sim_config.DEVICE, sim_config.QRR3_RESULTS_LOG)
    )



def quantized_rank_red_mixed_client_fn(context: Context) -> Client:
    net = utils.Net().to(sim_config.DEVICE)
    partition_id = context.node_config["partition-id"]

    return quantized_rank_red_client.QuantizedRankRedClient(partition_id, net, sim_config.TRAINLOADERS[partition_id], sim_config.QRR3_STORAGE[partition_id], sim_config.NUM_CLIENTS,
                                         sim_config.NUM_BITS, sim_config.P_MIXED[partition_id], sim_config.DEVICE).to_client()

def quantized_rank_red_mixed_server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=sim_config.NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=quantized_rank_red.QuantizedRankRed(copy.deepcopy(NET), sim_config.NUM_CLIENTS, sim_config.NUM_ROUNDS, sim_config.LEARNING_RATE, sim_config.NUM_BITS, sim_config.P_MIXED,
                                  sim_config.TESTLOADER, sim_config.DEVICE, sim_config.QRR3_RESULTS_LOG)
    )



def main():
    backend_config = {"client_resources": {"num_cpus": 2}}
    if sim_config.DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_gpus": 1, "num_cpus": 1}}
    
    # SGD
    client = ClientApp(client_fn=sgd_client_fn)
    server = ServerApp(server_fn=sgd_server_fn)
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=sim_config.NUM_CLIENTS,
        backend_config=backend_config,
    )

    # SLAQ
    client = ClientApp(client_fn=slaq_client_fn)
    server = ServerApp(server_fn=slaq_server_fn)
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=sim_config.NUM_CLIENTS,
        backend_config=backend_config,
    )
    sim_config.SLAQ_CLIENT_STORAGE = None

    # # Two way SLAQ
    # client = ClientApp(client_fn=two_way_slaq_client_fn)
    # server = ServerApp(server_fn=two_way_slaq_server_fn)
    # run_simulation(
    #     server_app=server,
    #     client_app=client,
    #     num_supernodes=sim_config.NUM_CLIENTS,
    #     backend_config=backend_config,
    # )
    # sim_config.TWO_WAY_SLAQ_CLIENT_STORAGE = None

    # # QuantizedRedRank1
    # client = ClientApp(client_fn=quantized_rank_red1_client_fn)
    # server = ServerApp(server_fn=quantized_rank_red1_server_fn)
    # run_simulation(
    #     server_app=server,
    #     client_app=client,
    #     num_supernodes=sim_config.NUM_CLIENTS,
    #     backend_config=backend_config,
    # )
    # # QuantizedRedRank2
    # client = ClientApp(client_fn=quantized_rank_red2_client_fn)
    # server = ServerApp(server_fn=quantized_rank_red2_server_fn)
    # run_simulation(
    #     server_app=server,
    #     client_app=client,
    #     num_supernodes=sim_config.NUM_CLIENTS,
    #     backend_config=backend_config,
    # )
    # sim_config.QRR2_STORAGE = None
    # # QuantizedRedRank3
    # client = ClientApp(client_fn=quantized_rank_red3_client_fn)
    # server = ServerApp(server_fn=quantized_rank_red3_server_fn)
    # run_simulation(
    #     server_app=server,
    #     client_app=client,
    #     num_supernodes=sim_config.NUM_CLIENTS,
    #     backend_config=backend_config,
    # )


    # QuantizedRedRank_Mixed
    client = ClientApp(client_fn=quantized_rank_red_mixed_client_fn)
    server = ServerApp(server_fn=quantized_rank_red_mixed_server_fn)
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=sim_config.NUM_CLIENTS,
        backend_config=backend_config,
    )

    plot_results([sim_config.SGD_RESULTS_LOG, sim_config.SLAQ_RESULTS_LOG, sim_config.QRR3_RESULTS_LOG], ["SGD", "SLAQ", "QRR"])


if __name__ == "__main__":
    main()