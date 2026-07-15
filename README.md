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

## 1.data  collection

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
## 2.Visualising and Analising the Data

## Importing The Libraries

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
## read the dataset

The project utilizes the following libraries:

* **Pandas** – For data loading, cleaning, transformation, and analysis.
* **NumPy** – For numerical computations and array operations.
* **Matplotlib** – For creating charts and visualizing data patterns.
* **Seaborn** – For advanced statistical data visualization.
* **Scikit-learn** – For data preprocessing, model training, prediction, and evaluation.
* **Joblib/Pickle** – For saving and loading trained machine learning models.
* **Warnings** – For handling and suppressing unnecessary warning messages.
![image alt](https://github.com/user-attachments/assets/99617e6a-a955-407a-a0ea-fa534ae727c2)
These libraries collectively support the complete workflow, from data preprocessing and exploratory data analysis to model building, evaluation, and deployment, ensuring accurate and efficient credit card approval predictions.
## Univariate Analysis

Univariate Analysis examines each feature in the dataset independently to understand its distribution and characteristics. In this project, it is used to analyze applicant attributes such as Occupation Type, Income, Education Level, Family Status, and Age.

Key tasks performed include:

Analyzing the frequency and distribution of individual features.
Detecting missing values, outliers, and class imbalance.
Summarizing numerical features using statistical measures.
Visualizing categorical and numerical data using Matplotlib and Seaborn.

Common functions used:

value_counts() – Counts the frequency of categorical values.
describe() – Generates summary statistics for numerical features.
sns.countplot() – Visualizes categorical feature distributions.
sns.histplot() and sns.boxplot() – Display numerical feature distributions and outliers.

## source code
# Frequency of occupation types
app['OCCUPATION_TYPE'].value_counts()

# Count plot for occupation type
sns.countplot(x='OCCUPATION_TYPE', data=app)
plt.xticks(rotation=90)
plt.show()

# Histogram for annual income
sns.histplot(app['AMT_INCOME_TOTAL'], bins=30, kde=True)
plt.show()
 
# Box plot for annual income
sns.boxplot(x=app['AMT_INCOME_TOTAL'])
plt.show()

This analysis provides a better understanding of the dataset, supports effective preprocessing and feature selection, and improves the performance of the Credit Card Approval Prediction model.

### Multivariate Analysis

Multivariate Analysis is used to examine the relationships between multiple features in the dataset simultaneously. In the **Credit Card Approval Prediction** project, it helps identify how applicant attributes such as **Age, Annual Income, Employment Status, Education Level, and Family Members** interact with each other.

**Key tasks performed:**

* Analyze relationships between multiple features.
* Identify feature correlations and dependencies.
* Detect redundant features and hidden patterns.
* Support feature selection for model training.

**Common functions used:**

* `corr()` – Calculates the correlation matrix.
* `sns.heatmap()` – Visualizes correlations between numerical features.
* `pairplot()` – Displays pairwise relationships between variables.

Multivariate analysis improves data understanding, enhances feature selection, and contributes to better machine learning model performance.
For GitHub, you don't need to explain every value in the heatmap. A short professional description is enough.

### Multivariate Analysis

The correlation heatmap below shows the relationships between numerical features in the Credit Card Approval dataset. It helps identify positive and negative correlations, detect feature dependencies, and understand how variables interact with each other before model training. Most features have weak correlations, while **`DAYS_BIRTH`** and **`DAYS_EMPLOYED`** show a relatively strong negative correlation, indicating an inverse relationship between age and employment days. This analysis supports feature selection, reduces redundancy, and improves machine learning model performance.

**Code Used:**

```python
plt.figure(figsize=(8,6))
sns.heatmap(app.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
For GitHub, keep it short and focused. Here's a professional version:

### Descriptive Analysis

Descriptive Analysis is used to summarize the statistical characteristics of the dataset and gain insights into numerical features before model training.

**Key tasks performed:**

* Summarized numerical features using statistical measures.
* Analyzed data distribution and variability.
* Identified minimum, maximum, mean, median, and standard deviation.
* Detected potential outliers and inconsistencies.

**Common function used:**

* `describe()` – Generates statistical summaries, including count, mean, standard deviation, minimum, maximum, and quartile values.

This analysis provides a clear understanding of the dataset, supports effective data preprocessing, and improves feature selection for the Credit Card Approval Prediction model.

![image alt](https://github.com/user-attachments/assets/01ae144a-7f42-4708-9d50-c1b4e84e8ece)
## 3.Data preprocessing
## Drop Duplicate features

## Removing Duplicate Records

Removing duplicate records is an essential data preprocessing step to ensure that each applicant is represented only once in the dataset. Duplicate entries can negatively impact data quality and reduce the accuracy of the machine learning model.

### Key Tasks Performed

* Identified duplicate records in the dataset.
* Removed duplicate rows using the Pandas `drop_duplicates()` function.
* Retained only unique applicant records for further analysis and model training.

### Common Function Used

```python
df.drop_duplicates()
```

or

```python
df.drop_duplicates(subset='Applicant_ID', keep='first')
```

Removing duplicate records improves data quality, ensures dataset consistency, and enhances the reliability of the Credit Card Approval Prediction model.
## handling missing values
For GitHub, use this concise version:

## Handling Missing Values

Handling missing values is an important preprocessing step to ensure the dataset is complete and suitable for machine learning. Missing values are identified and analyzed before model training to improve data quality.

### Key Tasks Performed

* Identified missing values in each feature.
* Calculated the total number and percentage of missing values.
* Removed unnecessary columns containing missing data.
* Verified that the cleaned dataset contains no missing values.

### Common Functions Used

```python
df.isnull().sum()
```

```python
df.isnull().mean()
```

```python
df.drop(columns=['OCCUPATION_TYPE'], inplace=True)
```

Handling missing values improves data quality, ensures reliable preprocessing, and prepares the dataset for feature selection, model training, and accurate credit card approval prediction.
For GitHub, keep it brief and focused on the preprocessing steps.

## Data Cleaning and Feature Transformation

Data Cleaning and Feature Transformation prepare the dataset for machine learning by removing inconsistencies, transforming features, and converting categorical data into numerical values.

### Key Tasks Performed

* Removed unnecessary columns to reduce data complexity.
* Converted negative values in `DAYS_BIRTH` and `DAYS_EMPLOYED` to positive values using `abs()`.
* Created new features to improve data representation.
* Encoded categorical features such as **Housing Type**, **Income Type**, **Education Type**, and **Family Status** into numerical values.
* Processed credit history data by grouping records using the applicant **ID**.
* Generated new features such as **Open Month**, **End Month**, and **Credit Window** from `MONTHS_BALANCE`.
* Analyzed the `STATUS` column to represent applicant payment behavior.

### Common Functions Used

```python
drop()
abs()
map()
groupby()
merge()
```
This preprocessing step improves data quality, enhances feature representation, and prepares the dataset for efficient model training and accurate credit card approval prediction.
For GitHub, you can summarize it like this:
## Feature engineering
## Label Encoding and Dataset Merging

The **`STATUS`** column is transformed from multiple payment categories into **binary classes** to simplify the credit card approval prediction task. Applicants with a good repayment history are labeled as **Approved (1)**, while those with overdue payments or poor credit history are labeled as **Not Approved (0)**.

### Key Tasks Performed

* Converted multi-class payment status into binary labels.
* Assigned **Approved (1)** for good repayment behavior.
* Assigned **Not Approved (0)** for poor repayment behavior.
* Merged applicant information with credit history using the **Applicant ID**.
* Created a unified dataset for machine learning model training.

### Common Functions Used

```python
map()
merge()
groupby()
```

This process simplifies the classification problem, integrates applicant and credit history data, and prepares the final dataset for accurate credit card approval prediction.
## Handlinng Categorial values
For GitHub, use this concise version:

## Label Encoding

Label Encoding is used to convert categorical features into numerical values so they can be processed by machine learning algorithms. The **`LabelEncoder`** class from Scikit-learn transforms each unique category into a unique integer value.

### Key Tasks Performed

* Converted categorical features into numerical labels.
* Encoded text-based values using the `fit_transform()` method.
* Prepared the dataset for machine learning model training.
* Preserved category information without assigning any ranking or priority.

### Common Functions Used

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Column_Name'] = le.fit_transform(df['Column_Name'])
```
![image alt]()
Label encoding enables machine learning models to process categorical data efficiently and improves the overall performance of the Credit Card Approval Prediction model.

## 4.module building

## Logistic Regression

Logistic Regression is used as a binary classification algorithm to predict whether a credit card application is **Approved** or **Rejected** based on applicant information.


### Key Steps

* Initialize the `LogisticRegression()` model.
* Train the model using the `fit()` method.
* Generate predictions using the `predict()` method.
* Compare predicted values with actual test data.

### Model Evaluation

* **Confusion Matrix** – Evaluates correct and incorrect predictions.
* **Classification Report** – Measures **Precision**, **Recall**, **F1-Score**, and **Accuracy**.
![image alt](https://github.com/user-attachments/assets/c15c0d52-2ce3-4051-a744-ae6a45f95f77)

Logistic Regression provides a simple, fast, and interpretable baseline model for the Credit Card Approval Prediction system, making it well-suited for binary classification tasks.
Here's a concise, GitHub-friendly version:

## Random Forest

Random Forest is a supervised machine learning algorithm used to classify credit card applications as **Approved** or **Rejected**. It builds multiple decision trees and combines their predictions using **majority voting**, resulting in improved accuracy and reduced overfitting.

### Key Steps

* Initialize the `RandomForestClassifier()` model.
* Train the model using the `fit()` method.
* Generate predictions using the `predict()` method.
* Evaluate the model using test data.

### Model Evaluation

* **Confusion Matrix** – Measures correct and incorrect predictions.
* **Classification Report** – Provides **Precision**, **Recall**, **F1-Score**, and **Accuracy**.
![image alt](https://github.com/user-attachments/assets/b77048c4-a73b-443b-9cba-f1dc47b941e2)

Random Forest delivers high prediction accuracy, handles large datasets efficiently, and is well-suited for credit card approval classification tasks.
Here's a concise, GitHub-friendly version:

## Decision Tree

Decision Tree is a supervised machine learning algorithm used to classify credit card applications as **Approved** or **Rejected**. It makes predictions by creating a tree-like structure of decision rules based on applicant features.

### Key Steps

* Initialize the `DecisionTreeClassifier()` model.
* Train the model using the `fit()` method.
* Generate predictions using the `predict()` method.
* Evaluate the model using test data.

### Model Evaluation

* **Confusion Matrix** – Measures correct and incorrect predictions.
* **Classification Report** – Provides **Precision**, **Recall**, **F1-Score**, and **Accuracy**.

## source code:
def d_tree(xtrain, xtest, ytrain, ytest):
    dt = DecisionTreeClassifier()
    dt.fit(xtrain, ytrain)
    ypred = dt.predict(xtest)

    print("***** Decision Tree Classifier *****")
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))

Decision Tree is simple, interpretable, and effective for handling both numerical and categorical data, making it a suitable model for credit card approval prediction.


## 5.Application building

## Building HTML pages

## Flask Web Application

The Credit Card Approval Prediction System is deployed using **Flask**, providing a simple and interactive web interface for real-time predictions. The application connects the frontend with the trained machine learning model to process user inputs and generate prediction results.

### Project Files

* **`home.html`** – Landing page with project overview and navigation.
* **`index.html`** – Form for entering applicant details.
* **`result.html`** – Displays the credit card approval prediction.
* **`app.py`** – Flask backend that manages routing, loads the trained model, processes user input, and generates predictions.

### Features

* Interactive web interface built with HTML and Flask.
* Loads the trained machine learning model using **Pickle/Joblib**.
* Accepts applicant details through a web form.
* Performs real -time credit card approval prediction.
* Displays the prediction result (Approved/Rejected) on the result page.

This Flask application enables users to interact with the machine learning model through a simple, responsive, and user-friendly interface.

![image alt](https://github.com/user-attachments/assets/088f7481-2447-43b0-83cf-b654515fcc5a)

## Building the python script
For GitHub, you can use this concise version:

## Building the Server-Side Script Using Flask and Pickle

The server-side application is developed using **Flask** to connect the frontend with the trained machine learning model. The saved model is loaded using **Pickle**, allowing predictions to be generated without retraining.

### Key Features

* Loads the trained model using **Pickle**.
* Handles routing between web pages using **Flask**.
* Collects user input through `request.form`.
* Processes input data and generates predictions using the model's `predict()` method.
* Displays the prediction result on the **result.html** page.

This server-side implementation enables real-time credit card approval prediction through a simple, fast, and user-friendly web application.

![image alt](https://github.com/user-attachments/assets/e9d81938-f85d-41d6-8b6c-943c1333285d)

### Run the application

## Running the Flask Application

The Flask application is executed locally to test the Credit Card Approval Prediction system and verify the integration between the frontend, backend, and trained machine learning model.

### Steps to Run

1. Open **Anaconda Prompt** or **Command Prompt**.
2. Navigate to the project directory containing `app.py`.
3. Start the Flask server using:

   ```bash
   python app.py
   ```
4. Open the generated local URL (e.g., `http://127.0.0.1:5000/`) in a web browser.

### Result
![image alt](https://github.com/user-attachments/assets/0173022a-ec29-4dba-bcec-f589a150c318)
* Access the application through the web interface.
* Enter applicant details in the prediction form.
* Submit the form to generate a prediction.
* View the final credit card approval result (**Approved** or **Rejected**) on the result page.

This step confirms that the Flask application, trained model, and frontend interface are successfully integrated and functioning correctly.

- 
## Conclusion

The Credit Card Approval Prediction System demonstrates the effective use of machine learning to automate the credit card approval process, reducing manual effort and improving decision-making accuracy. By analyzing applicant information such as income type, employment status, family status, housing type, and credit history, the system predicts whether a credit card application is likely to be approved or rejected.

The project implements a complete machine learning workflow, including data collection, preprocessing, exploratory data analysis, feature engineering, categorical encoding, model training, testing, and performance evaluation. Multiple classification algorithms—Logistic Regression, Decision Tree, Random Forest, and XGBoost—are trained and compared to identify the best-performing model based on prediction accuracy and reliability.

The selected model is integrated into a Flask web application, providing a simple and interactive interface for users and banking professionals to enter applicant details and receive instant approval predictions. Additionally, integration with IBM Watson Machine Learning enables cloud deployment, allowing the system to deliver scalable, secure, and real-time prediction services.

Overall, this project highlights the practical application of machine learning in the banking and financial sector. It enhances the efficiency of the credit approval process, minimizes human error, supports faster decision-making, and provides valuable hands-on experience in data science, web development, cloud deployment, and financial risk assessment.
