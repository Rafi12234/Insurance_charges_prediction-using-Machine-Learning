import gradio as gr
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_charges(age, sex, bmi, children, smoker, region):
    # Encode categorical variables
    sex_encoded = 1 if sex == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0
    
    # Encode region
    region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
    region_encoded = region_map[region]
    
    # Create bmi_category
    if bmi < 18.4:
        bmi_category = 'Underweight'
    elif bmi < 24.9:
        bmi_category = 'Normal weight'
    elif bmi < 29.9:
        bmi_category = 'Overweight'
    else:
        bmi_category = 'Obesity'
    
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex_encoded,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_encoded,
        "region": region_encoded,
        "bmi_category": bmi_category
    }])
    
    prediction = model.predict(input_df)
    return f"Predicted Insurance Charges: ${prediction[0]:.2f}"

inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=27.5),
    gr.Number(label="Children", value=0),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
]

app = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs="text",
    title="Insurance Charges Prediction (Random Forest)",
    description="Enter customer details to predict insurance charges."
)

app.launch(share=True)