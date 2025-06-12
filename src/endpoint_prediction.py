from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

if __name__ == "__main__":

    # Replace with your actual endpoint name
    endpoint_name = "inference-image-2025-06-12-16-13-37-695"

    # Create the predictor
    predictor = Predictor(
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    # Prepare your payload
    payload = {
        "instances": [
            {"ticker_symbol": "AAPL", "n_days": 30},
            {"ticker_symbol": "GOOGL", "n_days": 15},

        ]
    }

    # Make the prediction
    result = predictor.predict(payload)
    print(result)