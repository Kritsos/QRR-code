import matplotlib.pyplot as plt
import os
from typing import List

class ResultsLog():
    def __init__(self):
        self.metrics = {}


    def log(self, metric, value):
        if self.metrics.get(metric, None) is None:
            self.metrics[metric] = []

        self.metrics[metric].append(value)


def plot_results(logs: List[ResultsLog], labels: List[str]):
    os.makedirs("figs", exist_ok=True)

    # loss vs iteration
    plt.figure(1)
    plt.title("Loss vs Iteration")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["iteration"], logs[i].metrics["loss"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_1.png")

    # loss vs communications
    plt.figure(2)
    plt.title("Loss vs Communications")
    plt.xlabel("Number of communications")
    plt.ylabel("Loss")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["communications"], logs[i].metrics["loss"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_2.png")

    # loss vs bits
    plt.figure(3)
    plt.title("Loss vs Bits")
    plt.xlabel("Number of bits")
    plt.ylabel("Loss")
    plt.xscale("log")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["bits"], logs[i].metrics["loss"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_3.png")

    # gradient vs iteration
    plt.figure(4)
    plt.title("Gradient vs Iteration")
    plt.xlabel("Number of iterations")
    plt.ylabel("Gradient L2 Norm")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["iteration"][1:], logs[i].metrics["gradient"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_4.png")

    # gradient vs communications
    plt.figure(5)
    plt.title("Gradient vs Communications")
    plt.xlabel("Number of communications")
    plt.ylabel("Gradient L2 Norm")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["communications"][1:], logs[i].metrics["gradient"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_5.png")

    # gradient vs bits
    plt.figure(6)
    plt.title("Gradient vs Bits")
    plt.xlabel("Number of bits")
    plt.ylabel("Gradient L2 Norm")
    plt.xscale("log")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["bits"][1:], logs[i].metrics["gradient"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_6.png")

    # accuracy vs iteration
    plt.figure(7)
    plt.title("Accuracy vs Iteration")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["iteration"], logs[i].metrics["accuracy"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_7.png")

    # accuracy vs communications
    plt.figure(8)
    plt.title("Accuracy vs Communications")
    plt.xlabel("Number of communications")
    plt.ylabel("Accuracy")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["communications"], logs[i].metrics["accuracy"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_8.png")

    # accuracy vs bits
    plt.figure(9)
    plt.title("Accuracy vs Bits")
    plt.xlabel("Number of bits")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    for i in range(len(logs)):
        plt.plot(logs[i].metrics["bits"], logs[i].metrics["accuracy"], label=labels[i])
    plt.legend()
    plt.savefig("figs/Figure_9.png")

    s = ""
    for i in range(len(logs)):
        s += f"{labels[i]}:\n"
        for metric, vals in logs[i].metrics.items():
            s += f"{metric}: {vals[-1]}\n"
        s += "\n"
    
    with open("results.txt", "w") as f:
        f.write(s)
    print(s)

    plt.show()