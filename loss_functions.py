import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, output, teacher):
        pass

    @abstractmethod
    def differentiation(self, output, teacher):
        pass

class MSE(LossFunction):
    def __init__(self):
        pass

    def __call__(self, output, teacher):
        return 1/2 * np.sum((output - teacher) ** 2) / output.shape[0]
    
    def differentiation(self, output, teacher):
        return output - teacher
    