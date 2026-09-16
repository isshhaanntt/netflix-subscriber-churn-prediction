# Netflix Subscriber Churn Prediction using Logistic Regression

## 📌 Project Overview

This project uses **Machine Learning** to predict whether a Netflix subscriber is likely to **churn (cancel their subscription)** or **stay subscribed**.

The project implements a **Logistic Regression classification model** because it is simple, transparent, and easy to explain. The model assigns a coefficient to each feature, allowing us to understand which factors increase or decrease churn risk.

> **Note:** Netflix's real subscriber data is private. Therefore, this project uses a **realistic synthetic dataset** created specifically for this classroom case study.

---

## 🎯 Objective

The main objective of this project is to build a machine learning model that can:

* Predict subscriber churn
* Identify factors associated with churn
* Analyze subscriber engagement
* Understand the impact of payment failures
* Evaluate model performance
* Demonstrate how machine learning can support customer retention

---

## 🤖 Machine Learning Model

### Logistic Regression

**Logistic Regression** is a supervised machine learning algorithm used for classification problems.

In this project:

* `0` = Subscriber Stayed
* `1` = Subscriber Churned

Logistic Regression was selected because:

* It is easy to understand
* It is computationally efficient
* It works well for binary classification
* Its coefficients can be interpreted
* It is suitable for demonstrating business decision-making with machine learning

---

## 📊 Dataset

The notebook generates a synthetic dataset containing **3,000 subscriber records**.

Each subscriber contains the following information:

| Feature              | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `tenure_months`      | Number of months the subscriber has been subscribed       |
| `weekly_watch_hours` | Average hours watched per week                            |
| `logins_per_month`   | Number of times the subscriber logs in each month         |
| `support_tickets`    | Support tickets raised during the last 90 days            |
| `payment_failures`   | Failed payments during the last 90 days                   |
| `plan_tier`          | Basic, Standard, or Premium subscription                  |
| `churn`              | Target variable indicating whether the subscriber churned |

---

## 🔄 Project Workflow

The project follows these main steps:

```text
Synthetic Dataset
       ↓
Data Exploration
       ↓
Data Preprocessing
       ↓
One-Hot Encoding
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Logistic Regression
       ↓
Model Prediction
       ↓
Model Evaluation
       ↓
Feature Importance Analysis
       ↓
Live Churn Prediction
```

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

---

## 📚 Libraries Used

The project uses the following Python libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

## ⚙️ Data Preprocessing

Before training the model, the dataset is prepared using several steps.

### 1. One-Hot Encoding

The `plan_tier` categorical variable is converted into numerical features.

```python
df_model = pd.get_dummies(
    df,
    columns=['plan_tier'],
    drop_first=True
)
```

### 2. Train-Test Split

The data is divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 3. Feature Scaling

`StandardScaler` is used to scale numerical features before model training.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 🧠 Model Training

The Logistic Regression model is trained using the processed training data.

```python
model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)
```

The model learns relationships between subscriber characteristics and the churn label.

---

## 📈 Model Evaluation

The trained model is evaluated using the test dataset.

The project includes:

### Accuracy

Measures the overall percentage of correct predictions.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows:

* Correctly predicted stayed subscribers
* Correctly predicted churned subscribers
* Subscribers incorrectly classified as churned
* Churned subscribers incorrectly classified as staying

---

## 🔍 Feature Analysis

One important part of the project is analyzing the model's coefficients.

Each feature receives a **model weight**.

Generally:

* **Positive coefficient** → pushes the prediction toward higher churn risk
* **Negative coefficient** → pushes the prediction toward lower churn risk

The notebook visualizes these weights using a bar chart.

The model's learned patterns indicate that factors such as **payment failures and lower engagement** can increase churn risk, while **longer tenure and Premium subscription** can reduce the predicted risk in this synthetic dataset.

---

## 🧪 Live Churn Prediction

The notebook also includes a function that allows users to enter subscriber information and receive a churn prediction.

Example:

```python
predict_churn(
    tenure_months=3,
    weekly_watch_hours=1.5,
    logins_per_month=3,
    support_tickets=2,
    payment_failures=2,
    plan_tier='Basic'
)
```

The model returns:

```text
Predicted churn probability: XX.X%
Prediction: LIKELY TO CHURN
```

A second example demonstrates a highly engaged and loyal Premium subscriber.

---

## 💼 Business Application

A churn prediction model can help a subscription-based business identify customers who may be at higher risk of leaving.

Potential business actions could include:

* Retention offers
* Promotional discounts
* Free upgrade periods
* Proactive customer support
* Personalized recommendations
* Payment issue resolution
* Engagement campaigns

The purpose is to identify potential risk early enough for the business to take appropriate retention action.

---

## ⚠️ Important Limitation

This project uses **synthetic data**, not actual Netflix customer data.

Therefore:

* The results should not be interpreted as actual Netflix churn statistics.
* The model is intended for learning and demonstration.
* A real-world system would require genuine customer data.
* Real deployment would also require appropriate privacy, security, and governance practices.

---

## 📁 Repository Structure

```text
netflix-subscriber-churn-prediction/
│
├── netflix_churn_logistic_regression.ipynb
│
└── README.md
```

---

## 🚀 How to Run

### Option 1 — Google Colab

1. Open the `.ipynb` notebook.
2. Upload it to Google Colab.
3. Run the cells sequentially.
4. Review the dataset, charts, model results, and predictions.

### Option 2 — Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/your-username/netflix-subscriber-churn-prediction.git
```

Open the notebook:

```bash
jupyter notebook netflix_churn_logistic_regression.ipynb
```

Run the cells from top to bottom.

---

## 🎓 Learning Outcomes

Through this project, you can understand:

* Basics of supervised machine learning
* Binary classification
* Logistic Regression
* Synthetic data generation
* Data preprocessing
* One-hot encoding
* Feature scaling
* Train-test splitting
* Model evaluation
* Confusion matrices
* Classification reports
* Model coefficients
* Churn prediction
* Business applications of machine learning

---

## 🔮 Future Improvements

This project could be extended by:

* Using a real anonymized customer dataset
* Comparing Logistic Regression with Random Forest and other classifiers
* Performing hyperparameter tuning
* Adding ROC-AUC analysis
* Creating an interactive dashboard
* Adding customer segmentation
* Building a retention recommendation system
* Deploying the model as a web application

---

## 👩‍💻 Project Type

**Machine Learning | Classification | Customer Churn Analysis | Business Analytics**

## 📌 Keywords

`Python` `Machine Learning` `Logistic Regression` `Netflix` `Churn Prediction` `Customer Analytics` `Classification` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Data Science`
