from collections import deque

class ClientStorage():
    def __init__(self, D):
        self.timer = 0
        self.prev_quant_grad = None
        self.prev_weights = [None for _ in range(D + 1)]
        self.prev_weights = deque(self.prev_weights, maxlen=D+1)
        self.prev_quant_error = 0
        self.prev_quant_weights = None