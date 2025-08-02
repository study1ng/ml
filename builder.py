import tomllib
from activation_functions import Sigmoid, Identity
from layers import FullyConnectedLayer
from optimizers import GradientDescent, Momentum, RMSProp, Adam
from loss_functions import MSE
from model import Model
from preprocessor import Preprocessor
from preprocessor_methods import Standardization
from trainer import OnlineTrainer, FullBatchTrainer, MiniBatchTrainer, Trainer
import mlflow

class Builder:  
    _CONTENTMAP = {
        "FullyConnectedLayer": FullyConnectedLayer,
        "Sigmoid": Sigmoid,
        "Identity": Identity,
        "GradientDescent": GradientDescent,
        "Momentum": Momentum,
        "MSE": MSE,
        "Standardization": Standardization,
        "OnlineTrainer": OnlineTrainer,
        "FullBatchTrainer": FullBatchTrainer,
        "MiniBatchTrainer": MiniBatchTrainer,
        "RMSProp": RMSProp,
        "Adam": Adam
    }

    def __init__(self, config_path):
        mlflow.log_artifact(config_path)
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        self.config = self.recursive_preprocess(config=self.config)
        
    def recursive_preprocess(cls, config):
        config = cls.preprocess(config)
        for k, v in config.items():
            if isinstance(v, dict):
                config[k] = cls.recursive_preprocess(v)
            if isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], dict):
                        v[i] = cls.recursive_preprocess(v[i])
        return config

    @classmethod
    def preprocess(cls, config):
        config = config.copy()
        if "log" in config.keys():
            for k, v in config["log"].items():
                mlflow.log_param(k, config[v])
        config = {k: v for k, v in config.items() if k != "log"}
        for k, v in config.items():
            if v.__hash__ != None and v in cls._CONTENTMAP and k != "class":
                config[k] = cls._CONTENTMAP[config[k]]()
        return config

    @classmethod
    def parse_config(cls, config):
        cls = cls._CONTENTMAP[config["class"]]
        config = {k: v for k, v in config.items() if k != "class"}
        return cls(**config)

    @classmethod
    def parse_layer_config(cls, config, input_size, output_size):
        cls = cls._CONTENTMAP[config["class"]]
        config = {k: v for k, v in config.items() if k != "class"}
        if config["input_size"] == "input":
            config["input_size"] = input_size
        if config["output_size"] == "output":
            config["output_size"] = output_size
        return cls(**config)

    def build_model(self, input_size, output_size) -> Model:
        config = self.config["experiment"]["trainer"]["model"]
        layers = []
        for layer_config in config["layers"]:
            layers.append(self.parse_layer_config(layer_config, input_size, output_size))
        optimizer = self.parse_config(config["optimizer"])
        loss_func = self.parse_config(config["loss_function"])
        return Model(layers, optimizer, loss_func)

    def build_preprocessor(self) -> Preprocessor:
        config = self.config["experiment"]["preprocessor"]
        methods = config["methods"]
        pmethods = []
        for method in methods:
            pmethods.append(self.parse_config(method))
        return Preprocessor(pmethods)

    def build_trainer(self, input_size, output_size) -> Trainer:
        config = self.config["experiment"]["trainer"]
        model = self.build_model(input_size, output_size)
        cls = self._CONTENTMAP[config["class"]]
        return cls(model, **{k: v for k, v in config.items() if k != "class" and k != "model"})

    @property
    def expr_config(self):
        return {k: v for k, v in self.config["experiment"].items() if not isinstance(v, dict)}

