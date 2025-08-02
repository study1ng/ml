import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """Abstract base class for loss functions.

    This class defines the interface for functions that measure the
    discrepancy between the model's output and the ground truth. Any new loss
    function must implement methods to compute the loss value itself and its
    derivative, which serves as the initial gradient for backpropagation.
    """

    @abstractmethod
    def __init__(self):
        """Initializes the loss function."""
        pass

    @abstractmethod
    def __call__(self, output: np.ndarray, teacher: np.ndarray) -> float:
        """Computes the loss value.

        Args:
            output: The predictions from the model.
            teacher: The ground truth labels.

        Returns:
            A single scalar value representing the computed loss.
        """
        pass

    @abstractmethod
    def differentiation(self, output: np.ndarray, teacher: np.ndarray) -> np.ndarray:
        """Computes the derivative of the loss function.

        This computes the derivative with respect to the model's output,
        which is the first gradient in the backpropagation chain.

        Args:
            output: The predictions from the model.
            teacher: The ground truth labels.

        Returns:
            An array of the same shape as `output`, representing the
            gradient of the loss with respect to the model's output.
        """
        pass

class MSE(LossFunction):
    def __init__(self):
        pass

    def __call__(self, output, teacher):
        return 1/2 * np.sum((output - teacher) ** 2) / output.shape[0]
    
    def differentiation(self, output, teacher):
        return output - teacher
    