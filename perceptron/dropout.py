import random
from typing import List


class Dropout:
    def __init__(self, rate: float):
        self.rate = rate

        self.scaled_mask: List[float] = None

    def forward_pass(self, inputs: List[float]) -> List[float]:
        inputs_length = len(inputs)
        to_mask = random.sample(
            population=range(inputs_length),
            k=int(inputs_length * self.rate),
        )
        scale = 1 - self.rate
        self.scaled_mask = [
            0 if index in to_mask else 1 / scale for index in range(inputs_length)
        ]

        outputs = [inp * mask for inp, mask in zip(inputs, self.scaled_mask)]
        return outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        return [grad * mask for grad, mask in zip(outputs_gradients, self.scaled_mask)]
