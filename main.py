import numpy as np
from sklearn import datasets
import sklearn
from itertools import repeat
import mlflow

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
