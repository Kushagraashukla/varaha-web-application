# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:17:56 2025
@author: OMEN
"""

import pandas as pd
import streamlit as st
from joblib import load
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# ---- Your custom transformer ----
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# ---- Custom transformer ----
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {}
    def fit(self, X, y=None):
        for c in self.cols:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[c].apply(lambda x: x.split(',') if isinstance(x, str) else []))
            self.encoders[c] = mlb
        return self
    def transform(self, X):
        X = X.copy()
        for c, mlb in self.encoders.items():
            arr = mlb.transform(X[c].apply(lambda x: x.split(',') if isinstance(x, str) else []))
            new_cols = [f"{c}_{cls}" for cls in mlb.classes_]
            X = pd.concat([X.drop(columns=[c]),
                           pd.DataFrame(arr, columns=new_cols, index=X.index)],
                          axis=1)
        return X

# ---- Custom function ----
def drop_cols(df):
    low_impact = ['How Long TV PC Daily Hour', 'How Long Internet Daily Hour']
    return df.drop(columns=low_impact)

# ---------------- Load the bundled model + pipeline ----------------
def load_model_pipeline(path):
    bundle = load(path)
    return bundle["model"], bundle["pipeline"]

# ---------------- Collect user inputs ----------------
def get_user_inputs():
    sex          = st.selectbox("Sex", ["male", "female"])
    body_type    = st.selectbox("Body Type", ["obese","overweight","underweight","normal"])
    diet         = st.selectbox("Diet", ["omnivore","vegetarian","vegan","pescatarian"])
    shower       = st.selectbox("How Often Shower",
                                ["daily","less frequently","more frequently","twice a day"])
    heating      = st.selectbox("Heating Energy Source",
                                ["electricity","natural gas","wood","coal"])
    transport    = st.selectbox("Transport", ["public","private","walk/bicycle"])
    vehicle_type = st.selectbox("Vehicle Type",
                                ["petrol","diesel","electric","hybrid","lpg","unknown"])
    social       = st.selectbox("Social Activity", ["never","sometimes","often"])
    grocery      = st.number_input("Monthly Grocery Bill", min_value=0)
    air_freq     = st.selectbox("Frequency of Traveling by Air",
                                ["never","rarely","frequently","very frequently"])
    vehicle_km   = st.number_input("Vehicle Monthly Distance Km", min_value=0)
    bag_size     = st.selectbox("Waste Bag Size", ["small","medium","large","extra large"])
    bag_count    = st.number_input("Waste Bag Weekly Count", min_value=0)
    tv_hour      = st.number_input("How Long TV PC Daily Hour", min_value=0)
    new_clothes  = st.number_input("How Many New Clothes Monthly", min_value=0)
    internet_hr  = st.number_input("How Long Internet Daily Hour", min_value=0)
    energy_eff   = st.selectbox("Energy efficiency", ["Yes","No","Sometimes"])
    recycling    = st.multiselect("Recycling", ["Paper","Plastic","Glass","Metal"])
    cooking      = st.multiselect("Cooking With",
                                  ["Stove","Oven","Microwave","Grill","Airfryer"])
    
    # Convert inputs into a DataFrame
    input_df = pd.DataFrame([{
        "Sex": sex,
        "Body Type": body_type,
        "Diet": diet,
        "How Often Shower": shower,
        "Heating Energy Source": heating,
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Social Activity": social,
        "Monthly Grocery Bill": grocery,
        "Frequency of Traveling by Air": air_freq,
        "Vehicle Monthly Distance Km": vehicle_km,
        "Waste Bag Size": bag_size,
        "Waste Bag Weekly Count": bag_count,
        "How Long TV PC Daily Hour": tv_hour,
        "How Many New Clothes Monthly": new_clothes,
        "How Long Internet Daily Hour": internet_hr,
        "Energy efficiency": energy_eff,
        "Recycling": ",".join(recycling),      # convert list to string
        "Cooking_With": ",".join(cooking)      # convert list to string
    }])
    
    return input_df

# ---------------- Preprocess the input ----------------
def preprocess_input(input_df, preprocess_pipe):
    return preprocess_pipe.transform(input_df)

# ---------------- Make prediction ----------------
def predict_carbon_footprint(X_processed, model):
    return model.predict(X_processed)[0]

# ---------------- Main function ----------------
def main():
    st.title("🌱 Carbon Footprint Predictor")
    
    # Load model and pipeline
    model, preprocess_pipe = load_model_pipeline(
        r"C:\Users\OMEN\OneDrive\Documents\machine learning\ML projects\carbonfootprintpredictor\cccarbon_model.joblib"
    )
    
    # Get inputs
    input_df = get_user_inputs()
    
    # Button trigger
    if st.button("Predict"):
        try:
            # Preprocess & predict
            X_processed = preprocess_input(input_df, preprocess_pipe)
            pred = predict_carbon_footprint(X_processed, model)
            st.success(f"🌿 Estimated Carbon Footprint: {pred:.2f}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# ---------------- Run the app ----------------
if __name__ == "__main__":
    main()
