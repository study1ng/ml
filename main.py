import numpy as np
from sklearn import datasets
import sklearn
from itertools import repeat
import mlflow
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def differentiation(self, x):
        pass

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (np.exp(-x) + 1)

    def differentiation(self, x):
        return x * (1 - x)

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        # 自身の重み行列と入力をもとに次の層に渡す行列を計算する
        pass

    @abstractmethod
    def backward(self, dout):
        # 次の層の誤差信号を受け取り, 自身の重み行列による誤差の微分を計算する.  
        pass

    @classmethod
    def to2D(cls, x):
        if len(x.shape) == 1:
            return x[:, np.newaxis]
        return x


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
        self.input_size = input_size + 1 # bias
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = np.random.randn(self.input_size * self.output_size).reshape((self.output_size, self.input_size))
        self.dw = None
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = np.append(x, 1)
        self.input = self.to2D(self.input)
        self.output = self.activation_function(self.weights @ self.input)
        self.output = self.to2D(self.output)
        return self.output
    
    def backward(self, dout):
        delta = dout * self.activation_function.differentiation(self.output)
        self.dw = delta @ self.input.T
        return self.weights[:, :-1].T @ delta

class Optimizer(ABC):
    @abstractmethod
    def update(self, layers: list[Layer]):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layers: list[Layer]):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.dw

class LossFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, output, teacher):
        pass

    @abstractmethod
    def differentiation(self, output, teacher):
        pass

class MSE(LossFunction):
    def __init__(self):
        pass

    def __call__(self, output, teacher):
        return 1/2 * np.sum((output - teacher) ** 2)
    
    def differentiation(self, output, teacher):
        return output - teacher
    
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


with mlflow.start_run():
    iris = datasets.load_iris()
    inputs = iris.data
    teachers = iris.target
    ave = inputs.mean()
    std = inputs.std()
    inputs = (inputs - ave) / std

    # one-hot
    teachers = np.eye(3)[teachers]
    train_inputs, test_inputs, train_teachers, test_teachers = sklearn.model_selection.train_test_split(inputs, teachers, test_size=0.2, random_state = 0)

    # learning rate
    eta = 0.01
    epochs = 500
    mlflow.log_param("eta", eta)
    mlflow.log_param("epochs", epochs)

    HiddenLayer = FullyConnectedLayer(train_inputs.shape[1], 16, Sigmoid())
    OutputLayer = FullyConnectedLayer(16, train_teachers.shape[1], Sigmoid())
    optimizer = SGD(eta)
    model = Model([HiddenLayer, OutputLayer], optimizer, MSE)

    errors = []

    for epoch in range(epochs):
        error_sum = 0
        for i, t in zip(train_inputs, train_teachers):
            model.expect(i)
            error_sum += model.learn(t)
        errors.append(error_sum / len(train_inputs))
        mlflow.log_metric("training_loss", error_sum / len(train_inputs), step=epoch)
            

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(errors)
    fig.savefig("error.png")
    mlflow.log_artifact("error.png")
    mlflow.log_metrics({"error": errors[-1]})
    np.savez("weights.npz", model.layers[0].weights, model.layers[1].weights)
    mlflow.log_artifact("weights.npz")

    o = []

    # test
    for i, t in zip(test_inputs, test_teachers):
        out = model.expect(i)
        E = model.loss_func(out, t)
        print(E)
        o.append((np.argmax(out, axis=0), np.argmax(t, axis=0)))

    # visualize
    a = np.zeros((3, 3))
    for o in o:
        a[o[0], o[1]] += 1
    print(a)
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(a, annot=True, fmt=".0f", ax=ax)
    fig.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
