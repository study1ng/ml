import numpy as np
from abc import ABC, abstractmethod
from layers import Layer

class Optimizer(ABC):
    @abstractmethod
    def update(self, layers: list[Layer]):
        pass

class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers: list[Layer]):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.dw

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def update(self, layers: list[Layer]):
        if self.velocity is None:
            self.velocity = [np.zeros_like(layer.weights) for layer in layers]
        for i in range(len(layers)):
            self.velocity[i] = self.momentum_rate * self.velocity[i] - self.learning_rate * layers[i].dw
            layers[i].weights += self.velocity[i]
