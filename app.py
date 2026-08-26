import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0"
)

artifact = joblib.load("model/telco_churn_model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]


class Customer(BaseModel):
    customerID: str
    
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
def predict(customer: Customer):

    data = customer.model_dump()

    customer_id = data.pop("customerID")

    df = pd.DataFrame([data])

    probability = model.predict_proba(df)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "customerID": customer_id,
        "churn": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }