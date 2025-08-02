from abc import ABC, abstractmethod
import numpy as np
import mlflow
import random



class Trainer(ABC):
    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs
        mlflow.log_param("epochs", epochs)
    
    @abstractmethod
    def train(self, train_inputs, train_teachers):
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
        mlflow.log_param("batch_size", batch_size)
    
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