# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset.

The project includes EDA, preprocessing pipelines, model comparison, hyperparameter tuning, a FastAPI prediction API, a simple web interface, and Docker containerization.

## Dataset

Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

## Machine Learning

The following classification models were compared:

- Logistic Regression
- Support Vector Classifier
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

Logistic Regression was selected as the final model after model comparison and hyperparameter tuning.

## Final Model Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| No Churn | 0.89 | 0.76 | 0.82 | 1033 |
| Churn | 0.53 | 0.74 | 0.61 | 374 |

| Overall Metric | Value |
|---|---:|
| Accuracy | 0.75 |
| Macro Avg F1 | 0.72 |
| Weighted Avg F1 | 0.76 |

The decision threshold was tuned to approximately `0.58` to improve the precision-recall balance for churn detection.

## Project Structure

```text
telco-customer-churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
├── frontend/
│   └── index.html
├── model/
│   └── telco_churn_model.pkl
├── notebooks/
│   └── telco_churn_analysis.ipynb
├── app.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Run with Docker

Make sure Docker Desktop is installed and running.

Build the Docker image:

```bash
docker build -t telco-churn-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 telco-churn-api
```

Open the web interface:

```text
http://127.0.0.1:8000
```

## Run the Notebook

To run the notebook and reproduce the analysis/model training:

```bash
pip install -r requirements-dev.txt
```

Then open:

```text
notebooks/telco_churn_analysis.ipynb
```