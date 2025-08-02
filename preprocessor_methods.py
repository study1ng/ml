from abc import ABC, abstractmethod
import numpy as np

class PreprocessorMethod(ABC):
    """Abstract base class for data preprocessing methods.

    This class defines the interface for preprocessing steps like standardization
    or normalization.
    """

    @abstractmethod
    def fit(self, train_inputs: np.ndarray, train_teachers: np.ndarray):
        """Learns the parameters required for the transformation.

        This method should only be called on the training dataset. It computes
        the necessary statistics (e.g., mean, standard deviation) from the
        training data and stores them as internal attributes of the object.

        Args:
            train_inputs: The input features of the training data.
            train_teachers: The target labels of the training data.
        """
        pass

    @abstractmethod
    def transform(self, inputs: np.ndarray, teachers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies the learned transformation to the data.

        This method uses the parameters learned by the `fit` method to
        transform the given data. It should be applied to both the training
        and test datasets.

        Args:
            inputs: The input features to be transformed.
            teachers: The target labels to be transformed.

        Returns:
            A tuple containing the transformed inputs and teachers.
        """
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