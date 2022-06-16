import random
from typing import List


class Dropout:
    def __init__(self, rate: float):
        self.rate = rate

        self.scaled_mask: List[float] = None

    def forward_pass(self, inputs: List[float], training: bool = False) -> List[float]:

        if not training:
            return inputs

        idx_to_mask = random.sample(range(len(inputs)), k=int(len(inputs) * self.rate))
        scale = 1.0 - self.rate
    
        self.scaled_mask = [
            0.0 if idx in idx_to_mask else 1.0 / scale for idx in range(len(inputs))
        ]

        return [inp * mask for inp, mask in zip(inputs, self.scaled_mask)]

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        return [grad * mask for grad, mask in zip(outputs_gradients, self.scaled_mask)]
