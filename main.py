import numpy as np
from sklearn import datasets
import sklearn
import mlflow
from activation_functions import Sigmoid
from layers import FullyConnectedLayer
from optimizers import SGD
from model import Model
from loss_functions import MSE

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
