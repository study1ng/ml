import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):    
    """Abstract base class for activation functions.

    This class defines the interface for all activation functions used in the
    neural network. Any new activation function should inherit from this class
    and implement both the `__call__` and `differentiation` methods.
    """
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:        
        """Applies the activation function to the input.

        This method performs the forward pass of the activation function.

        Args:
            x: The input numpy array, typically the output of a linear layer.

        Returns:
            The numpy array after applying the activation function.
        """
        pass

    @abstractmethod
    def differentiation(self, y: np.ndarray) -> np.ndarray:
        """Computes the derivative of the activation function.

        Note:
            For computational efficiency, the input `y` to this method is
            often the *output* of the forward pass (`__call__`), not the
            original input. For example, for the sigmoid function,
            the derivative y * (1 - y) is calculated from its output y.

        Args:
            x: The numpy array at which to compute the derivative.

        Returns:
            The derivative of the activation function.
        """
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

    def differentiation(self, y):
        return np.ones_like(y)