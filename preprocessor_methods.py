from abc import ABC, abstractmethod

class PreprocessorMethod(ABC):
    @abstractmethod
    def fit(self, train_inputs, train_teachers):
        pass

    @abstractmethod
    def transform(self, train_inputs, train_teachers):
        pass

class Standardization(PreprocessorMethod):
    def __init__(self):
        pass

    def fit(self, train_inputs, train_teachers):
        self.input_ave = train_inputs.mean(axis=0)
        self.input_std = train_inputs.std(axis=0)
        self.teacher_ave = train_teachers.mean(axis=0)
        self.teacher_std = train_teachers.std(axis=0)

    def transform(self, input, teacher):
        input = (input - self.input_ave) / self.input_std
        teacher = (teacher - self.teacher_ave) / self.teacher_std
        return input, teacher