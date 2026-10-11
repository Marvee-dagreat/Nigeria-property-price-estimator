from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "property_price_model.pkl"

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]


# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Nigerian Property Price Prediction API",
    description="Predicts property prices using our trained XGBoost model.",
    version="1.0.0"
)


app.add_middleware(                                   # <- new lines (right after app = FastAPI())
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. Define the input data
# --------------------------------------------------

class PropertyInput(BaseModel):

    market_status: str = "Available"

    property_type: str = Field(..., alias="Type")

    district: str = Field(..., alias="District")

    state: str = Field(..., alias="State")

    servicing: str = "Unknown"

    bedrooms: float | None = None

    bathrooms: float | None = None

    toilets: float | None = None

    parking_spaces: float | None = None

    added_year: int

    added_month: int


# --------------------------------------------------
# 4. Health/root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Nigerian Property Price Prediction API is running."
    }


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_price(property_data: PropertyInput):

    # Convert the request into a dictionary
    data = property_data.model_dump(by_alias=True)

    # Build dataframe with the ORIGINAL feature names
    df_input = pd.DataFrame([{
        "Market Status": data["market_status"],
        "Type": data["Type"],
        "District": data["District"],
        "State": data["State"],
        "Servicing": data["servicing"],

        "Bedrooms": data["bedrooms"],
        "Bathrooms": data["bathrooms"],
        "Toilets": data["toilets"],
        "Parking Spaces": data["parking_spaces"],

        "Added Year": data["added_year"],
        "Added Month": data["added_month"]
    }])

    # --------------------------------------------------
    # Create the same engineered features used in training
    # --------------------------------------------------

    df_input["Total Rooms"] = (
        df_input["Bedrooms"]
        + df_input["Bathrooms"]
        + df_input["Toilets"]
    )

    df_input["Bathrooms_per_Bedroom"] = (
        df_input["Bathrooms"]
        / df_input["Bedrooms"].replace(0, np.nan)
    )

    df_input["Toilets_per_Bedroom"] = (
        df_input["Toilets"]
        / df_input["Bedrooms"].replace(0, np.nan)
    )

    df_input["Parking_per_Bedroom"] = (
        df_input["Parking Spaces"]
        / df_input["Bedrooms"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # Preprocess exactly like training
    # --------------------------------------------------

    processed_input = preprocessor.transform(df_input)

    # --------------------------------------------------
    # Predict log(price)
    # --------------------------------------------------

    predicted_log_price = model.predict(processed_input)

    # Convert log prediction back to Naira
    predicted_price = np.expm1(predicted_log_price)[0]

    return {
        "predicted_price": round(float(predicted_price), 2),
        "currency": "NGN",
        "formatted_price": f"₦{predicted_price:,.2f}"
    }