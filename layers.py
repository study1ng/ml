import numpy as np
from activation_functions import ActivationFunction
from abc import ABC, abstractmethod

class Layer(ABC):
    """Abstract base class for a layer in the neural network.

    This class defines the fundamental interface for all layers. Each layer
    must be able to perform a forward pass (to compute its output) and a
    backward pass (to compute gradients for its parameters and to propagate
    the gradient backwards).
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass of the layer.

        Args:
            x: Input data from the previous layer.

        Returns:
            The output of this layer.
        """
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Performs the backward pass of the layer (backpropagation).

        This method has two main responsibilities:
        1. Compute the gradients for the layer's own parameters (e.g.,
           weights and biases) and store them internally.
        2. Propagate the gradient backwards to the previous layer.

        Args:
            dout: The gradient of the loss with respect to the output of this
                  layer. This is the "downstream" gradient coming from the
                  next layer in the network.

        Returns:
            The gradient of the loss with respect to the input of this layer.
        """
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
        self.actual_input_size = input_size
        self.input_size = input_size + 1 # bias
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = np.random.randn(self.input_size * self.output_size).reshape((self.input_size, self.output_size))
        self.dw = None
        self.input = None
        self.output = None
    
    def forward(self, x):
        x = x
        self.input = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.output = self.activation_function(self.input @ self.weights)
        return self.output
    
    def backward(self, dout):
        delta = dout * self.activation_function.differentiation(self.output)
        self.dw = self.input.T @ delta
        self.dw /= self.input.shape[0]
        return delta @ self.weights.T[:, :-1]