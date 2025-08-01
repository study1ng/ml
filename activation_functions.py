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

    def differentiation(self, x):
        return x * (1 - x)
