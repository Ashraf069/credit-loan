import argparse
import os

try:
    from ibm_watson_machine_learning import APIClient
except ImportError:
    APIClient = None

MODEL_FILENAME = "model_pipeline.joblib"


def get_wml_client(api_key: str, url: str):
    if APIClient is None:
        raise RuntimeError(
            "The ibm-watson-machine-learning package is not installed."
        )

    config = {"apikey": api_key, "url": url}
    return APIClient(config)


def deploy_model(wml_client, space_id: str, model_path: str, model_name: str):
    wml_client.set.default_space(space_id)

    metadata = {
        "name": model_name,
        "type": "scikit-learn_1.0",
        "description": "Credit card approval model pipeline",
    }

    model_props = wml_client.repository.store_model(
        model=model_path,
        meta_props=metadata,
    )

    model_uid = wml_client.repository.get_model_uid(model_props)
    deployment_props = {
        "name": f"{model_name}-deployment",
        "output_data_schema": {
            "type": "dict",
        },
    }

    deployment = wml_client.deployments.create(
        artifact_uid=model_uid,
        meta_props=deployment_props,
    )
    return deployment


def main(model_path: str, model_name: str, api_key: str, url: str, space_id: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_model.py and provide the generated pipeline."
        )

    client = get_wml_client(api_key, url)
    deployment = deploy_model(client, space_id, model_path, model_name)
    print("Deployment created successfully.")
    print(deployment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the trained model to IBM Watson Machine Learning.")
    parser.add_argument(
        "--model",
        default=MODEL_FILENAME,
        help="Path to the trained model pipeline file.",
    )
    parser.add_argument(
        "--name",
        default="credit-card-approval-model",
        help="Name for the Watson deployment and model artifact.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("WML_API_KEY"),
        help="IBM Watson ML API key. Can also be set via WML_API_KEY.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("WML_URL"),
        help="IBM Watson ML service URL. Can also be set via WML_URL.",
    )
    parser.add_argument(
        "--space-id",
        default=os.environ.get("WML_SPACE_ID"),
        help="IBM Watson ML space ID. Can also be set via WML_SPACE_ID.",
    )
    args = parser.parse_args()

    if not args.api_key or not args.url or not args.space_id:
        raise SystemExit(
            "WML_API_KEY, WML_URL, and WML_SPACE_ID are required. Set them via environment variables or command line."
        )

    main(args.model, args.name, args.api_key, args.url, args.space_id)
