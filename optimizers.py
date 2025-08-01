import numpy as np
from abc import ABC, abstractmethod
from layers import Layer

class Optimizer(ABC):
    @abstractmethod
    def update(self, layers: list[Layer]):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layers: list[Layer]):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.dw
