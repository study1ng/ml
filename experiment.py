from model import Model
import sklearn
import numpy as np
import mlflow
import time


class Experiment:
    def __init__(self, builder, dataset, test_portion, random_state=None):
        self.builder = builder
        self.test_portion = test_portion
        self.random_state = random_state
        self.data = dataset.data
        self.target = dataset.target

    def preprocess(self):
        self.preprocessor = self.builder.build_preprocessor()
        train_inputs, test_inputs, train_teachers, test_teachers = sklearn.model_selection.train_test_split(self.data, self.target, test_size=self.test_portion, random_state=self.random_state)
        train_teachers = train_teachers[:, np.newaxis]
        test_teachers = test_teachers[:, np.newaxis]
        mlflow.log_param("train_size", train_inputs.shape[0])
        mlflow.log_param("test_size", test_inputs.shape[0])
        self.train_inputs, self.train_teachers = self.preprocessor.fit_transform(train_inputs, train_teachers)
        self.test_inputs, self.test_teachers = self.preprocessor.transform(test_inputs, test_teachers)
        input_size = self.train_inputs.shape[1]
        output_size = self.train_teachers.shape[1]
        self.trainer = self.builder.build_trainer(input_size, output_size)

    def process(self):
        self.trainer.train(self.train_inputs, self.train_teachers)

    def run(self) -> Model:
        self.preprocess()
        train_start_time = time.perf_counter()
        self.process()
        train_end_time = time.perf_counter()   
        mlflow.log_metric("training_time", train_end_time - train_start_time)
        return self.trainer.model
