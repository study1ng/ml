from layers import Layer
from optimizers import Optimizer
from loss_functions import LossFunction

class Model:
    def __init__(self, layers: list[Layer], optimizer: Optimizer, loss_func: LossFunction):
        self.layers = layers
        self.output = None
        self.optimizer = optimizer
        self.loss_func = loss_func


    def expect(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        return x

    def learn(self, teacher):
        loss = self.loss_func(self.output, teacher)
        error_signal = self.loss_func.differentiation(self.output, teacher)
        for layer in reversed(self.layers):
            error_signal = layer.backward(error_signal)
        self.optimizer.update(self.layers)
        return loss
    
    def loss(self, output, teacher):
        return self.loss_func(output, teacher)
    