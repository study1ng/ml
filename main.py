from sklearn import datasets
import mlflow
import time
from builder import Builder
from experiment import Experiment


with mlflow.start_run(nested=True):
    # load data
    diabetes = datasets.load_diabetes()
    inputs = diabetes.data
    teachers = diabetes.target

    # train
    builder = Builder("exprconfig.toml")
    experiment = Experiment(builder, diabetes, **builder.expr_config)
    model = experiment.run()

    model.save_weights("artifacts/weights.npz")

    # test
    test_start_time = time.perf_counter()
    out = model.expect(experiment.test_inputs)
    test_end_time = time.perf_counter()
    mlflow.log_metric("test_time", test_end_time - test_start_time)

    E = model.loss_func(out, experiment.test_teachers)
    mlflow.log_metric("test_loss", E)

    # visualize
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    fig, ax = plt.subplots()
    ax.scatter(out, experiment.test_teachers)
    fig.savefig("artifacts/test.png")
    mlflow.log_artifact("artifacts/test.png")
