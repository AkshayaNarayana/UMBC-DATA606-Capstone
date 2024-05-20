

import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Load the dataset
ds = pd.read_csv('jobs_postings.csv')

# Function to train ARIMA model
def train_arima_model(data):
    ds_train = data["Openings"][:len(data["Openings"]) - 30]
    auto_arima = pm.auto_arima(ds_train, stepwise=False, seasonal=False)
    return auto_arima

# Function to make predictions using ARIMA model
def predict_job_openings(model, test_data):
    if model is None:
        return None
    forecast = model.predict(n_periods=test_data)
    return forecast

# Load the trained ARIMA model
arima_model = train_arima_model(ds)

# Streamlit App
st.title('Job Openings Predictor')

# Date input from user
selected_date = st.date_input('Select a date')

# When user submits the date
if st.button('Predict'):
    # Calculate the number of days between the selected date and the last date in the dataset
    date_difference = (pd.to_datetime(selected_date) - pd.to_datetime(ds['Job Posting Date'].iloc[-1])).days

    if date_difference <= 0:
        st.write('Please select a future date.')
    else:
        # Make predictions for the selected date
        prediction = predict_job_openings(arima_model, date_difference)
        # Display the last prediction value
        if prediction is not None:
            last_prediction = list(prediction)[-1]  # Convert Series to list and get the last element
            st.write(f'Predicted number of job openings on {selected_date} is: {int(last_prediction)}')
        else:
            st.write('Model not trained or encountered an error.')
