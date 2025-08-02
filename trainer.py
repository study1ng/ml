from abc import ABC, abstractmethod
import numpy as np
import mlflow
from model import Model


class Trainer(ABC):
    """Abstract base class for training strategies.

    This class defines the interface for running a training loop. Different
    training strategies, such as online, mini-batch, or full-batch learning,
    should be implemented as subclasses.

    Attributes:
        model: The model to be trained.
        epochs: The total number of epochs for training.
    """
    def __init__(self, model: Model, epochs: int):
        """Initializes the Trainer.

        Args:
            model: The neural network model to be trained.
            epochs: The total number of epochs to train the model.
        """
        self.model = model
        self.epochs = epochs

    @abstractmethod
    def train(self, train_inputs: np.ndarray, train_teachers: np.ndarray):
        """Executes the full training loop.

        This method contains the logic for iterating over the training data
        for the specified number of epochs and updating the model's parameters.

        Args:
            train_inputs: The input features of the training data.
            train_teachers: The target labels of the training data.
        """
        pass

class OnlineTrainer(Trainer):
    def __init__(self, model, epochs):
        super().__init__(model, epochs)
    
    def train(self, train_inputs, train_teachers):
        for epoch in range(self.epochs):
            sum_of_loss = 0
            for i, t in zip(train_inputs, train_teachers):
                self.model.expect(i[np.newaxis, :])
                sum_of_loss += self.model.learn(t)
            mlflow.log_metric("training_loss", sum_of_loss / train_inputs.shape[0], step=epoch)

class FullBatchTrainer(Trainer):
    def __init__(self, model, epochs):
        super().__init__(model, epochs)

    def train(self, train_inputs, train_teachers):
        for epoch in range(self.epochs):
            self.model.expect(train_inputs)
            loss = self.model.learn(train_teachers)
            mlflow.log_metric("training_loss", loss, step=epoch)    

class MiniBatchTrainer(Trainer):
    def __init__(self, model, epochs, batch_size):
        super().__init__(model, epochs)
        self.batch_size = batch_size
    
    def train(self, train_inputs, train_teachers):
        for epoch in range(self.epochs):
            idx = np.arange(train_inputs.shape[0])
            np.random.shuffle(idx)
            train_inputs = train_inputs[idx]
            train_teachers = train_teachers[idx]
            sum_of_loss = 0
            for i in range(0, train_inputs.shape[0], self.batch_size):
                self.model.expect(train_inputs[i:min(i+self.batch_size, train_inputs.shape[0])])
                sum_of_loss += self.model.learn(train_teachers[i:min(i+self.batch_size, train_teachers.shape[0])]) * (min(i+self.batch_size, train_inputs.shape[0]) - i)
            mlflow.log_metric("training_loss", sum_of_loss / train_inputs.shape[0], step=epoch)