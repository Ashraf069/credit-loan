import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    _HAS_XGBOOST = False

NUMERIC_FEATURES = [
    "age",
    "income",
    "employment_duration_years",
    "existing_loan_balance",
    "credit_inquiries_last_6m",
    "num_credit_cards",
    "high_risk_flag",
]

CATEGORICAL_FEATURES = [
    "employment_type",
    "marital_status",
    "property_area",
    "payment_status_code",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "approved"

PAYMENT_STATUS_CODES = ["C0", "C1", "C2", "C3", "C4", "C5"]
EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "Unemployed", "Student"]
MARITAL_STATUSES = ["Single", "Married", "Divorced"]
PROPERTY_AREAS = ["Urban", "Semiurban", "Rural"]

def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_model_candidates(random_state=42):
    models = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
    }

    if _HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, n_jobs=-1)
    else:
        models["xgboost"] = GradientBoostingClassifier(random_state=random_state)

    return models


def compute_high_risk_flag(payment_status_code: str) -> int:
    return 1 if payment_status_code in {"C3", "C4", "C5"} else 0


def generate_synthetic_credit_data(n_samples=2500, random_state=42):
    rng = np.random.default_rng(random_state)

    def sample_choice(values, weights=None):
        return rng.choice(values, size=n_samples, p=weights)

    age = rng.integers(18, 71, size=n_samples)
    income = np.round(rng.normal(65000, 25000, size=n_samples).clip(12000, 220000), 2)
    employment_duration_years = rng.integers(0, 31, size=n_samples)
    existing_loan_balance = np.round(rng.normal(12000, 14000, size=n_samples).clip(0, 70000), 2)
    credit_inquiries_last_6m = rng.integers(0, 12, size=n_samples)
    num_credit_cards = rng.integers(0, 7, size=n_samples)
    employment_type = sample_choice(EMPLOYMENT_TYPES, weights=[0.55, 0.25, 0.15, 0.05])
    marital_status = sample_choice(MARITAL_STATUSES, weights=[0.45, 0.45, 0.10])
    property_area = sample_choice(PROPERTY_AREAS, weights=[0.40, 0.35, 0.25])
    payment_status_code = sample_choice(PAYMENT_STATUS_CODES, weights=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04])

    high_risk_flag = [compute_high_risk_flag(code) for code in payment_status_code]

    score = (
        (income / 20000)
        + (employment_duration_years / 5)
        - (existing_loan_balance / 10000)
        - (credit_inquiries_last_6m * 0.5)
        + np.array(high_risk_flag) * -2.5
        + np.where(num_credit_cards <= 4, 1.0, 0.0)
        + np.where((age >= 25) & (age <= 60), 1.0, 0.0)
    )

    probability = 1 / (1 + np.exp(-0.15 * (score - 2.5)))
    approved = (probability + rng.normal(0, 0.12, size=n_samples)) > 0.5

    data = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "employment_duration_years": employment_duration_years,
            "existing_loan_balance": existing_loan_balance,
            "credit_inquiries_last_6m": credit_inquiries_last_6m,
            "num_credit_cards": num_credit_cards,
            "employment_type": employment_type,
            "marital_status": marital_status,
            "property_area": property_area,
            "payment_status_code": payment_status_code,
            "high_risk_flag": high_risk_flag,
            "approved": approved.astype(int),
        }
    )
    return data


def build_prediction_dataframe(values: dict) -> pd.DataFrame:
    df = pd.DataFrame([values], columns=ALL_FEATURES)
    if "payment_status_code" in values:
        df["high_risk_flag"] = df["payment_status_code"].apply(compute_high_risk_flag)
    return df


def build_pipeline(model):
    preprocessor = build_preprocessor()
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])
