import sagemaker
import boto3
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)

#role = sagemaker.get_execution_role()
role = "arn:aws:iam::468155809742:role/service-role/AmazonSageMaker-ExecutionRole-20250609T234891"


from sagemaker.estimator import Estimator

#get the fir of the current file
import os
current_file_path = os.path.abspath(__file__)
import os.path as path
source_dir = os.path.join(path.dirname(current_file_path),"src")
source_path = os.path.join(source_dir, "timeseries_train.py")
estimator = Estimator(
    image_uri="468155809742.dkr.ecr.us-east-1.amazonaws.com/python/train-image:latest",
    entry_point= "timeseries_train.py",
    #entry_point="timeseries_train.py",
    source_dir=source_dir,
    role=role,
    instance_count=1,
    #instance_type="local",
    instance_type="ml.m5.large",
    output_path="s3://stock-market-data-udacity-flagship/output",
    base_job_name="timeline-model-hyperparam-tuning"
)

#estimator.fit(wait=True, logs=True)

# Uncomment the following lines if you want to use the PyTorch estimator directly

# estimator = PyTorch(
#     entry_point="pytorch_mnist.py",
#     role=role,
#     py_version='py36',
#     framework_version="1.8",
#     instance_count=1,
#     instance_type="ml.m5.large"
# )

hyperparameter_ranges = {
    "lag": IntegerParameter(1, 10),
    #"lr": ContinuousParameter(0.001, 0.1),
    #"batch-size": CategoricalParameter([32, 64, 128, 256, 512]),
}

objective_metric_name = "MAPE"
objective_type = "Minimize"
metric_definitions = [{"Name": "MAPE", "Regex": "Mean absolute percentage error: ([0-9\\.]+)%"}]

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_parallel_jobs=10,
    objective_type=objective_type,
    max_jobs = 10,
    strategy="Random",




)

tuner.fit(wait=False,verbose=True)

print(tuner.best_estimator().hyperparameters())