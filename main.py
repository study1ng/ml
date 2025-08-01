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

class HiddenLayer(Layer):
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
        if len(self.input) == 1:
            self.input = self.input[:, np.newaxis] # make 2D for batch process
        self.output = self.activation_function(self.weights @ self.input)
        return self.activation_function(self.weights @ self.input)
    
    def backward(self, dout):
        delta = dout * self.activation_function.differentiation(self.output)
        self.dw = delta @ self.input.T
        return self.weights[:-1].T @ delta
    
    @property
    def dw(self):
        return self.dw

class OutputLayer(HiddenLayer):
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
        if len(self.input) == 1:
            self.input = self.input[:, np.newaxis] # make 2D for batch process
        self.output = self.activation_function(self.weights @ self.input)
        return self.output
    
    def backward(self, error_signal):
        self.dw = error_signal @ self.input[:, np.newaxis].T
        return error_signal[:-1] # the last record of error_signal is about next layer's bias so remove it
        
    @property
    def dw(self):
        return self.dw

class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.differential = []
    
    def expect(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        return x

    def learn(self, teacher):
        error_signal = self.output - teacher
        for layer in reversed(self.layers):
            error_signal = layer.backward(error_signal)
        

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

    # hidden layer: 16 neurons, output layer: 16 neurons
    H = np.random.randn((inputs.shape[1] + 1) * 16).reshape((16, inputs.shape[1] + 1)) # bias
    O = np.random.randn((16 + 1) * teachers.shape[1]).reshape((teachers.shape[1], 16+1)) # bias

    def sigmoid(v):
        return 1 / (np.exp(-v) + 1)

    def sigmoid_differention(v):
        return v * (1 - v)

    # learning rate
    eta = 0.01
    epochs = 500
    mlflow.log_param("eta", eta)
    mlflow.log_param("epochs", epochs)

    errors = []

    for _ in range(epochs):
        esum = 0
        for i, t in zip(train_inputs, train_teachers):
            i = i.reshape((i.shape[0], 1))
            # forward propagation
            i = np.append(i, 1)
            mh = H @ i
            oh = sigmoid(mh)
            oh = np.append(oh, 1)
            mo = O @ oh
            oo = sigmoid(mo)
            e = oo - t
            E = 1/2 * (e @ e)
            esum += E
            # backpropagation
            do = e * oo * (1 - oo)
            # remove weights about bias
            do = do[:, np.newaxis]
            oh = oh[:, np.newaxis]
            eo = do @ oh.T
            dh = (O.T @ do)[:-1] * oh[:-1] * (1 - oh[:-1])
            i = i[:, np.newaxis]
            eh = dh @ i.T
            O -= eta * eo
            H -= eta * eh
        # print(esum / len(train_inputs))
        errors.append(esum / len(train_inputs))

    from matplotlib import pyplot as plt
    plt.plot(errors)
    plt.savefig("error.png")
    mlflow.log_artifact("error.png")
    mlflow.log_metrics({"error": errors[-1]})
    np.savez("weights.npz", H=H, O=O)
    mlflow.log_artifact("weights.npz")

    o = []

    # test
    for i, t in zip(test_inputs, test_teachers):
        i = i.reshape((i.shape[0], 1))
        # forward propagation
        i = np.append(i, 1)
        mh = H @ i
        oh = sigmoid(mh)
        oh = np.append(oh, 1)
        mo = O @ oh
        oo = sigmoid(mo)
        o.append((np.argmax(oo), np.argmax(t)))
        e = oo - t
        E = 1/2 * (e @ e)
        print(E)

    # visualize
    a = np.zeros((3, 3))
    for o in o:
        a[o[0], o[1]] += 1
    print(a)
    import seaborn as sns
    sns.heatmap(a, annot=True, fmt=".0f")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
