import numpy as np
from activation_functions import ActivationFunction
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        # 自身の重み行列と入力をもとに次の層に渡す行列を計算する
        pass

    @abstractmethod
    def backward(self, dout):
        # 次の層の誤差信号を受け取り, 自身の重み行列による誤差の微分を計算する.  
        pass

    @classmethod
    def to2D(cls, x):
        if len(x.shape) == 1:
            return x[:, np.newaxis]
        return x


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
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