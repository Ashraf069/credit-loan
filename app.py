import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

from model_utils import (
    ALL_FEATURES,
    EMPLOYMENT_TYPES,
    MARITAL_STATUSES,
    PROPERTY_AREAS,
    PAYMENT_STATUS_CODES,
    build_prediction_dataframe,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")


def create_app():
    app = Flask(__name__)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model pipeline not found. Run 'python train_model.py' first to create {MODEL_PATH}."
        )

    model_pipeline = joblib.load(MODEL_PATH)

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        form_values = {
            "age": 32,
            "income": 56000,
            "employment_duration_years": 5,
            "existing_loan_balance": 9000,
            "credit_inquiries_last_6m": 1,
            "num_credit_cards": 2,
            "employment_type": "Salaried",
            "marital_status": "Single",
            "property_area": "Urban",
            "payment_status_code": "C0",
        }

        if request.method == "POST":
            try:
                inputs = {
                    "age": int(request.form["age"]),
                    "income": float(request.form["income"]),
                    "employment_duration_years": int(request.form["employment_duration_years"]),
                    "existing_loan_balance": float(request.form["existing_loan_balance"]),
                    "credit_inquiries_last_6m": int(request.form["credit_inquiries_last_6m"]),
                    "num_credit_cards": int(request.form["num_credit_cards"]),
                    "employment_type": request.form["employment_type"],
                    "marital_status": request.form["marital_status"],
                    "property_area": request.form["property_area"],
                    "payment_status_code": request.form["payment_status_code"],
                }
                form_values.update(inputs)
                inputs_df = build_prediction_dataframe(inputs)
                probability = model_pipeline.predict_proba(inputs_df)[0][1]
                prediction = model_pipeline.predict(inputs_df)[0]

                result = {
                    "decision": "Approved" if prediction == 1 else "Rejected",
                    "probability": float(probability),
                }
            except Exception as exc:
                result = {"error": str(exc)}

        return render_template(
            "index.html",
            result=result,
            form_values=form_values,
            employment_types=EMPLOYMENT_TYPES,
            marital_statuses=MARITAL_STATUSES,
            property_areas=PROPERTY_AREAS,
            payment_status_codes=PAYMENT_STATUS_CODES,
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        request_data = request.get_json(force=True)
        if not request_data:
            return {"error": "Request body must be JSON with applicant feature values."}, 400

        inputs = {field: request_data.get(field) for field in ALL_FEATURES}
        missing = [field for field, value in inputs.items() if value is None]
        if missing:
            return {"error": f"Missing fields: {', '.join(missing)}"}, 400

        inputs["age"] = int(inputs["age"])
        inputs["income"] = float(inputs["income"])
        inputs["employment_duration_years"] = int(inputs["employment_duration_years"])
        inputs["existing_loan_balance"] = float(inputs["existing_loan_balance"])
        inputs["credit_inquiries_last_6m"] = int(inputs["credit_inquiries_last_6m"])
        inputs["num_credit_cards"] = int(inputs["num_credit_cards"])

        inputs_df = build_prediction_dataframe(inputs)
        probability = model_pipeline.predict_proba(inputs_df)[0][1]
        prediction = model_pipeline.predict(inputs_df)[0]

        return {
            "decision": "Approved" if prediction == 1 else "Rejected",
            "approval_probability": float(probability),
        }

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
