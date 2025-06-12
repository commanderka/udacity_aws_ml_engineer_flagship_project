# TODO: Deploy your model to an endpoint
import sagemaker
import os

from sagemaker.model import Model
from sagemaker.estimator import Estimator


if __name__ == "__main__":

    #train model first and then deploy it

    role = "arn:aws:iam::468155809742:role/service-role/AmazonSageMaker-ExecutionRole-20250609T234891"


    #get the fir of the current file
    import os
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    import os.path as path
    source_dir = os.path.join(path.dirname(current_file_path),"src")
    hyperparameters = {
        "lag": 10,
        "epochs": 2
    }
    estimator = Estimator(
        image_uri="468155809742.dkr.ecr.us-east-1.amazonaws.com/python/train-image:latest",
        entry_point= "timeseries_train.py",
        source_dir=source_dir,
        role=role,
        instance_count=1,
        #instance_type="local",
        instance_type="ml.m5.large",
        output_path="s3://stock-market-data-udacity-flagship/output",
        base_job_name="timeline-model-training",
        hyperparameters=hyperparameters
    )

    #estimator.fit(wait=True, logs=True)

    #deploy

    #get the model data from the latest training job

    latest_training_job_name = "timeline-model-training-2025-06-11-21-07-53-280"
    output_path = "s3://stock-market-data-udacity-flagship/output"
    #print(f"Latest training job name: {estimator.latest_training_job.job_name}")
    #print(f"Output path: {estimator.output_path}")
    #print(f"Model data path: {estimator.model_data}")
    #print(f"Model data: {estimator.model_data}")



    model_data = os.path.join(output_path,latest_training_job_name,"output","model.tar.gz")
    #model_data= "model.tar.gz"
    print(f"Model: {model_data}")

    darts_model = Model(
        image_uri="468155809742.dkr.ecr.us-east-1.amazonaws.com/python/inference-image:latest",
        model_data=model_data, 
        role=role, 
        #entry_point=os.path.join(current_file_dir,'inference.py'),
    )

    instance_type = "ml.t2.xlarge"
    predictor=darts_model.deploy(initial_instance_count=1, instance_type=instance_type)
    predictions = predictor.predict({"instances": [{"ticker_symbol": "AAPL", "n_days": 30}]})