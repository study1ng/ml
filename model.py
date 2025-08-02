from layers import Layer
from optimizers import Optimizer
from loss_functions import LossFunction
import json, mlflow, numpy as np

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
    
    def log(self):
        architecture = []
        for layer in self.layers:
            layer_info = {"class": layer.__class__.__name__}
            if hasattr(layer, 'input_size'):
                layer_info["input_size"] = layer.actual_input_size if hasattr(layer, 'input_size') else "N/A"
            if hasattr(layer, 'output_size'):
                layer_info["output_size"] = layer.output_size
            if hasattr(layer, 'activation_function'):
                layer_info["activation"] = layer.activation_function.__class__.__name__
            architecture.append(layer_info)

        architecture_path = "artifacts/architecture.json"

        with open(architecture_path, 'w') as f:
            json.dump(architecture, f, indent=4)

        mlflow.log_artifact(architecture_path)

    def save_weights(self, path):
        np.savez(path, *[layer.weights for layer in self.layers])
        mlflow.log_artifact(path)