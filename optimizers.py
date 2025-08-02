import numpy as np
from abc import ABC, abstractmethod
from layers import Layer

class Optimizer(ABC):
    """Abstract base class for optimization algorithms.

    This class defines the interface for all optimizers, which are responsible
    for updating the model's parameters based on the computed gradients.
    Subclasses must implement the `update` method, which encapsulates the
    specific logic of an optimization algorithm (e.g., SGD, Momentum).
    """

    @abstractmethod
    def update(self, layers: list[Layer]):
        """Performs a single optimization step to update model parameters.

        This method iterates through all layers of the model and updates their
        trainable parameters (e.g., weights and biases) according to the
        optimizer's specific update rule.

        It assumes that the gradients for each layer's parameters have already
        been computed and stored within each layer object (e.g., as `layer.dw`)
        by a preceding backward pass.

        Args:
            layers: A list of all layers in the model to be updated.
        """
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

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-12):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, layers: list[Layer]):
        if self.cache is None:
            self.cache = [np.zeros_like(layer.weights) for layer in layers]
        for i in range(len(layers)):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * layers[i].dw ** 2
            layers[i].weights -= self.learning_rate * layers[i].dw / (np.sqrt(self.cache[i]) + self.epsilon)
        
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, layers: list[Layer]):
        if self.m is None:
            self.m = [np.zeros_like(layer.weights) for layer in layers]
            self.v = [np.zeros_like(layer.weights) for layer in layers]
        
        self.t += 1

        for i in range(len(layers)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layers[i].dw
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * layers[i].dw ** 2
            mhat = self.m[i] / (1 - self.beta1 ** self.t)
            vhat = self.v[i] / (1 - self.beta2 ** self.t)
            layers[i].weights -= self.learning_rate * mhat / (np.sqrt(vhat) + self.epsilon)