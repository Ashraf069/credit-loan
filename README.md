# Credit Card Approval Prediction

This project automates credit card approval decisions using machine learning. It trains multiple classifiers on synthetic applicant data and exposes a Flask web interface for real-time approval predictions.

## Features
- Trains four classifiers: Logistic Regression, Random Forest, Decision Tree, and XGBoost.
- Automatically selects and saves the best-performing model.
- Flask app provides a user-friendly interface for single application predictions.
- Includes an IBM Watson Machine Learning deployment utility.
- Synthetic dataset generation simulates realistic applicant profiles.

- ## Entity relation ship diagram
- ![image alt](https://github.com/Ashraf069/credit-loan/blob/eb36367b1c5e47af9e7b1936ed30c1e6e74d680d/Entity%20Relationship%20Diagram/Credit%20Card%20Approval%20Prediction.png)
- 
## Prerequisites

| Prerequisite                | Official Link                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Python 3.10+                | [https://www.python.org/downloads/](https://www.python.org/downloads/)                                                     |
| Flask                       | [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)                                                   |
| Jupyter Notebook            | [https://jupyter.org/](https://jupyter.org/)                                                                               |
| Anaconda (optional)         | [https://www.anaconda.com/download](https://www.anaconda.com/download)                                                     |
| NumPy                       | [https://numpy.org/install/](https://numpy.org/install/)                                                                   |
| Pandas                      | [https://pandas.pydata.org/docs/getting_started/install.html](https://pandas.pydata.org/docs/getting_started/install.html) |
| Scikit-learn                | [https://scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html)                               |
| XGBoost                     | [https://xgboost.readthedocs.io/en/stable/install.html](https://xgboost.readthedocs.io/en/stable/install.html)             |
| Matplotlib                  | [https://matplotlib.org/stable/users/installing.html](https://matplotlib.org/stable/users/installing.html)                 |
| IBM Cloud                   | [https://cloud.ibm.com/](https://cloud.ibm.com/)                                                                           |
| IBM Watson Machine Learning | [https://www.ibm.com/products/watson-machine-learning](https://www.ibm.com/products/watson-machine-learning)               |
| Git                         | [https://git-scm.com/downloads](https://git-scm.com/downloads)                                                             |
| Visual Studio Code          | [https://code.visualstudio.com/](https://code.visualstudio.com/)    

## project flow
![image alt](https://github.com/Ashraf069/credit-loan/blob/374f261b39ceb9b27242b1d1da9e19e76b976b6f/work%20flow.png)

## data  collection

For Step 1: Data Collection, you should provide the dataset source used for training the Credit Card Approval Prediction model. The most commonly used dataset for this project is the Credit Card Approval Prediction dataset on Kaggle.

Dataset Download

Primary Dataset (Recommended)

Credit Card Approval Prediction (Kaggle):
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

This dataset includes:

Applicant demographic information
Income type
Employment status
Family status
Housing type
Education level
Occupation
Credit history
Payment status
Target label (derived through preprocessing)
Files Included

After downloading, you'll typically find:

application_record.csv – Applicant demographic and financial information
credit_record.csv – Credit history and payment records.

## - data collection workflow

Data Collection
      │
      ▼
Download Dataset from Kaggle
      │
      ▼
application_record.csv
credit_record.csv
      │
      ▼
Merge the datasets using ID
      │
      ▼
Clean Missing Values
      │
      ▼
Feature Engineering
      │
      ▼
Create Approval/Reject Labels
      │
      ▼
Train Machine Learning Models
      │
      ▼
Deploy Best Model using Flask

Dataset Description :

The project uses the Credit Card Approval Prediction dataset obtained from Kaggle. It consists of two CSV files: application_record.csv, which contains demographic and financial information of applicants, and credit_record.csv, which stores applicants' historical credit payment records. These datasets are merged using the applicant ID, preprocessed to remove inconsistencies, and transformed through feature engineering to create binary approval labels. The processed dataset is then used to train and evaluate multiple machine learning models for credit card approval prediction.

- 
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
## Visualising and Analising the Data

## 1.Importing The Libraries

Importing the required libraries is the initial step in the Credit Card Approval Prediction project. These libraries provide essential tools for data manipulation, visualization, preprocessing, machine learning model development, and performance evaluation. Using well-established Python libraries ensures efficient implementation, improves code readability, and accelerates the development process.

The project utilizes the following libraries:

Pandas – For data loading, cleaning, transformation, and analysis.
NumPy – For numerical computations and array operations.
Matplotlib – For creating charts and visualizing data patterns.
Seaborn – For advanced statistical data visualization.
Scikit-learn – For data preprocessing, model training, prediction, and evaluation.
Joblib/Pickle – For saving and loading trained machine learning models.
Warnings – For handling and suppressing unnecessary warning messages.

These libraries collectively support the complete workflow, from data preprocessing and exploratory data analysis to model building, evaluation, and deployment, ensuring accurate and efficient credit card approval predictions.
If you want a description for the **"Importing Required Libraries"** step in your GitHub project documentation, you can use something like this:

# Importing Required Libraries

Importing the required libraries is the initial step in the Credit Card Approval Prediction project. These libraries provide essential tools for data manipulation, visualization, preprocessing, machine learning model development, and performance evaluation. Using well-established Python libraries ensures efficient implementation, improves code readability, and accelerates the development process.
![image alt](https://github.com/user-attachments/assets/b77048c4-a73b-443b-9cba-f1dc47b941e2)

The project utilizes the following libraries:

* **Pandas** – For data loading, cleaning, transformation, and analysis.
* **NumPy** – For numerical computations and array operations.
* **Matplotlib** – For creating charts and visualizing data patterns.
* **Seaborn** – For advanced statistical data visualization.
* **Scikit-learn** – For data preprocessing, model training, prediction, and evaluation.
* **Joblib/Pickle** – For saving and loading trained machine learning models.
* **Warnings** – For handling and suppressing unnecessary warning messages.

These libraries collectively support the complete workflow, from data preprocessing and exploratory data analysis to model building, evaluation, and deployment, ensuring accurate and efficient credit card approval predictions.

### Sample Code

```python
# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model Saving
import joblib

# Warning Handling
import warnings
warnings.filterwarnings('ignore')
```

### 

> The project begins by importing essential Python libraries required for data analysis, preprocessing, visualization, machine learning model development, and evaluation. These libraries enable efficient handling of the credit card approval dataset, support the training of predictive models, and facilitate performance assessment to deliver accurate approval predictions.


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
- 
## Conclusion

The Credit Card Approval Prediction System demonstrates the effective use of machine learning to automate the credit card approval process, reducing manual effort and improving decision-making accuracy. By analyzing applicant information such as income type, employment status, family status, housing type, and credit history, the system predicts whether a credit card application is likely to be approved or rejected.

The project implements a complete machine learning workflow, including data collection, preprocessing, exploratory data analysis, feature engineering, categorical encoding, model training, testing, and performance evaluation. Multiple classification algorithms—Logistic Regression, Decision Tree, Random Forest, and XGBoost—are trained and compared to identify the best-performing model based on prediction accuracy and reliability.

The selected model is integrated into a Flask web application, providing a simple and interactive interface for users and banking professionals to enter applicant details and receive instant approval predictions. Additionally, integration with IBM Watson Machine Learning enables cloud deployment, allowing the system to deliver scalable, secure, and real-time prediction services.

Overall, this project highlights the practical application of machine learning in the banking and financial sector. It enhances the efficiency of the credit approval process, minimizes human error, supports faster decision-making, and provides valuable hands-on experience in data science, web development, cloud deployment, and financial risk assessment.
