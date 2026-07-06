# Credit Card Approval Prediction

This project automates credit card approval decisions using machine learning. It trains multiple classifiers on synthetic applicant data and exposes a Flask web interface for real-time approval predictions.

## Features
- Trains four classifiers: Logistic Regression, Random Forest, Decision Tree, and XGBoost.
- Automatically selects and saves the best-performing model.
- Flask app provides a user-friendly interface for single application predictions.
- Includes an IBM Watson Machine Learning deployment utility.
- Synthetic dataset generation simulates realistic applicant profiles.

## Setup
1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

If you want to deploy the model to IBM Watson Machine Learning, install the optional deployment requirements separately:

```powershell
pip install -r requirements-ibm.txt
```

## Training
Generate training data, train the candidate models, and save the best pipeline.

```powershell
python train_model.py
```

This creates `data/sample_credit_applications.csv` and `model_pipeline.joblib`.

## Running the Flask App

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## IBM Watson Machine Learning Deployment

Update `wml_deploy.py` with your IBM Watson credentials or set environment variables:
- `WML_API_KEY`
- `WML_URL`
- `WML_SPACE_ID`

Run:

```powershell
python wml_deploy.py --model model_pipeline.joblib --name "credit-card-approval"
```

## Project Structure
- `train_model.py`: Training pipeline and model selection.
- `model_utils.py`: Data generation, preprocessing, and model utilities.
- `app.py`: Flask application for predictions.
- `wml_deploy.py`: IBM Watson Machine Learning deployment helper.
- `templates/index.html`: Web UI template.

## Notes
- The application uses synthetic data when no real dataset is provided.
- XGBoost is included as a candidate model but falls back to a scikit-learn gradient boosting classifier if unavailable.
