import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from model_utils import (
    ALL_FEATURES,
    TARGET_COLUMN,
    build_model_candidates,
    build_pipeline,
    generate_synthetic_credit_data,
)


def select_best_model(models, X_train, y_train, X_test, y_test):
    best_name = None
    best_pipeline = None
    best_score = -1.0
    results = {}

    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_score)
        results[name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "model": model,
            "pipeline": pipeline,
        }

        if roc_auc > best_score:
            best_score = roc_auc
            best_name = name
            best_pipeline = pipeline

    return best_name, best_pipeline, results


def main(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "sample_credit_applications.csv")

    data = generate_synthetic_credit_data(n_samples=3200)
    data.to_csv(dataset_path, index=False)

    X = data[ALL_FEATURES]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_candidates = build_model_candidates(random_state=42)
    best_name, best_pipeline, results = select_best_model(
        model_candidates, X_train, y_train, X_test, y_test
    )

    print("Model comparison")
    for name, result in results.items():
        print(f"- {name}: accuracy={result['accuracy']:.4f}, roc_auc={result['roc_auc']:.4f}")

    print(f"\nBest model: {best_name}")

    model_path = os.path.join(output_dir, "model_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)
    with open(os.path.join(output_dir, "best_model.txt"), "w", encoding="utf-8") as handle:
        handle.write(best_name)

    print(f"Saved model pipeline to: {model_path}")
    print(f"Saved training dataset to: {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit card approval models and save the best pipeline.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated model and sample data files.",
    )
    args = parser.parse_args()
    main(args.output_dir)
