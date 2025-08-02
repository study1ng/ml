from sklearn import datasets
import mlflow
import time
from builder import Builder
from experiment import Experiment


with mlflow.start_run(nested=True):
    diabetes = datasets.load_diabetes()
    inputs = diabetes.data
    teachers = diabetes.target
    builder = Builder("exprconfig.toml")
    experiment = Experiment(builder, diabetes, **builder.expr_config)
    model = experiment.run()

    model.save_weights("artifacts/weights.npz")

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    # test
    test_start_time = time.perf_counter()
    out = model.expect(experiment.test_inputs)
    test_end_time = time.perf_counter()
    mlflow.log_metric("test_time", test_end_time - test_start_time)

    fig, ax = plt.subplots()
    ax.scatter(out, experiment.test_teachers)
    fig.savefig("artifacts/test.png")
    mlflow.log_artifact("artifacts/test.png")

    E = model.loss_func(out, experiment.test_teachers)
    mlflow.log_metric("test_loss", E)
