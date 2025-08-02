import tomllib
from activation_functions import Sigmoid, Identity
from layers import FullyConnectedLayer
from optimizers import GradientDescent, Momentum
from loss_functions import MSE
from model import Model
import mlflow

_CONTENTMAP = {
    "FullyConnectedLayer": FullyConnectedLayer,
    "Sigmoid": Sigmoid,
    "Identity": Identity,
    "GradientDescent": GradientDescent,
    "Momentum": Momentum,
    "MSE": MSE,
}

def preprocess(config):
    config = config.copy()
    if "log" in config.keys():
        for k, v in config["log"].items():
            mlflow.log_param(k, v)
    for k in config.keys():
        if config[k] in _CONTENTMAP and k != "class":
            config[k] = _CONTENTMAP[config[k]]()
    return {k: v for k, v in config.items() if k != "log"}

def parse_config(config):
    config = preprocess(config)
    cls = _CONTENTMAP[config["class"]]
    config = {k: v for k, v in config.items() if k != "class"}
    return cls(**config)

def parse_layer_config(config, input_size, output_size):
    config = preprocess(config)
    cls = _CONTENTMAP[config["class"]]
    config = {k: v for k, v in config.items() if k != "class"}
    if config["input_size"] == "input":
        config["input_size"] = input_size
    if config["output_size"] == "output":
        config["output_size"] = output_size
    return cls(**config)


def build_model_with_config(config_path, train_inputs, train_teachers) -> Model:
    mlflow.log_artifact(config_path)
    with open(config_path, "rb") as f:
        config = tomllib.load(f)["experiment"]["model"]
    layers = []
    input_size = train_inputs.shape[-1]
    output_size = train_teachers.shape[-1]
    for layer_config in config["layers"]:
        layers.append(parse_layer_config(layer_config, input_size, output_size))
    optimizer = parse_config(config["optimizer"])
    loss_func = parse_config(config["loss_function"])
    return Model(layers, optimizer, loss_func)
