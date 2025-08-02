import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def differentiation(self, x):
        pass

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (np.exp(-x) + 1)

    def differentiation(self, y):
        return y * (1 - y)

class Identity(ActivationFunction):
    """Identity function (does nothing)"""
    def __call__(self, x):
        return x

    def differentiation(self, x):
        return np.ones_like(x)