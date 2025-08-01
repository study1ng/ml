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
    loss_func = MSE()
    model = Model([HiddenLayer, OutputLayer], optimizer, loss_func)

    errors = []

    for epoch in range(epochs):
        model.expect(train_inputs)
        loss = model.learn(train_teachers)
        errors.append(loss)
        mlflow.log_metric("training_loss", loss, step=epoch)
            

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(errors)
    fig.savefig("artifacts/error.png")
    mlflow.log_artifact("artifacts/error.png")
    mlflow.log_metrics({"error": errors[-1]})
    np.savez("artifacts/weights.npz", model.layers[0].weights, model.layers[1].weights)
    mlflow.log_artifact("artifacts/weights.npz")

    # test
    out = model.expect(test_inputs)
    E = model.loss_func(out, test_teachers)
    mlflow.log_metric("test_loss", E)
    print(np.argmax(out, axis=1))
    print(np.argmax(test_teachers, axis=1))
    o = list(zip(np.argmax(out, axis=1), np.argmax(test_teachers, axis=1)))

    # visualize
    confusion_matrix = np.zeros((3, 3))
    for o in o:
        confusion_matrix[o[0], o[1]] += 1
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", ax=ax)
    fig.savefig("artifacts/confusion_matrix.png")
    mlflow.log_artifact("artifacts/confusion_matrix.png")
