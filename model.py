import numpy as np
from abc import ABC, abstractmethod
from layers import Layer
from optimizers import Optimizer
from loss_functions import LossFunction

class Model:
    def __init__(self, layers: list[Layer], optimizer: Optimizer, loss: LossFunction):
        self.layers = layers
        self.output = None
        self.optimizer = optimizer
        self.loss_func = loss()


    def expect(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        if len(self.output.shape) == 1:
            self.output = self.output[:, np.newaxis] # make 2D for batch process
        return x

    def learn(self, teacher):
        teacher = self.to2D(teacher)
        loss = self.loss_func(self.output, teacher)
        error_signal = self.loss_func.differentiation(self.output, teacher)
        for layer in reversed(self.layers):
            error_signal = layer.backward(error_signal)
        self.optimizer.update(self.layers)
        return loss
    
    def loss(self, output, teacher):
        return self.loss_func(output, teacher)
    
    @classmethod
    def to2D(cls, x):
        return Layer.to2D(x)

