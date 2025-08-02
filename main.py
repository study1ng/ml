import numpy as np
from sklearn import datasets
import sklearn
import mlflow
from config_parser import build_model_with_config
import time

with mlflow.start_run():
    diabetes = datasets.load_diabetes()
    inputs = diabetes.data
    teachers = diabetes.target

    train_inputs, test_inputs, train_teachers, test_teachers = sklearn.model_selection.train_test_split(inputs, teachers, test_size=0.2, random_state = 0)
    train_teachers = train_teachers[:, np.newaxis]
    test_teachers = test_teachers[:, np.newaxis]
    mlflow.log_param("train_size", train_inputs.shape[0])
    mlflow.log_param("test_size", test_inputs.shape[0])

    ave = train_inputs.mean(axis=0)
    std = train_inputs.std(axis=0)
    train_inputs = (train_inputs - ave) / std
    test_inputs = (test_inputs - ave) / std
    ave = train_teachers.mean(axis=0)
    std = train_teachers.std(axis=0)
    train_teachers = (train_teachers - ave) / std
    test_teachers = (test_teachers - ave) / std

    epochs = 2000
    mlflow.log_param("epochs", epochs)


    model = build_model_with_config("exprconfig.toml", train_inputs, train_teachers)
    model.log()
    errors = []
    
    train_start_time = time.perf_counter()
    for epoch in range(epochs):
        model.expect(train_inputs)
        loss = model.learn(train_teachers)
        errors.append(loss)
        mlflow.log_metric("training_loss", loss, step=epoch)
        
    model.save_weights("artifacts/weights.npz")

    train_end_time = time.perf_counter()
    mlflow.log_metric("training_time", train_end_time - train_start_time)

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(errors)
    fig.savefig("artifacts/error.png")
    mlflow.log_artifact("artifacts/error.png")
    mlflow.log_metrics({"error": errors[-1]})

    # test
    test_start_time = time.perf_counter()
    out = model.expect(test_inputs)
    test_end_time = time.perf_counter()
    mlflow.log_metric("test_time", test_end_time - test_start_time)

    for i, v in enumerate(out - test_teachers):
        mlflow.log_metric("test_error", v, step=i)

    fig, ax = plt.subplots()
    ax.scatter(out, test_teachers)
    fig.savefig("artifacts/test.png")
    mlflow.log_artifact("artifacts/test.png")


    E = model.loss_func(out, test_teachers)
    mlflow.log_metric("test_loss", E)
