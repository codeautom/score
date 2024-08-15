import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize label encoder and priority mappings
label_encoder = LabelEncoder()
label_encoder.fit(['Qualification', 'Needs Analysis', 'Meeting/Conference Call', 'RFP', 'Proposal/Price Quote', 'Negotiation/Review'])

priority_reverse_mapping = {1: 'Low', 2: 'Medium', 3: 'High'}

# Load the trained model
model_filename = 'client_priority_model.pkl'
model = joblib.load(model_filename)

# Streamlit app
st.title("Client Priority Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    
    # Extract the "Stage" column and filter out "Lost" and "Won" stages
    stage_data = data[['Stage']].dropna()
    stage_data = stage_data[~stage_data['Stage'].isin(['Lost', 'Won'])]
    
    # Encode the "Stage" column
    stage_data['Stage_Encoded'] = label_encoder.transform(stage_data['Stage'])
    
    # Prepare data for prediction
    X = stage_data[['Stage_Encoded']]
    
    # Predict client priorities
    predictions = model.predict(X)
    
    # Map numerical predictions back to priority levels
    stage_data['Predicted Priority'] = [priority_reverse_mapping[p] for p in predictions]
    
    # Merge predictions back into the original dataset
    data_with_predictions = data.merge(stage_data[['Stage', 'Predicted Priority']], on='Stage', how='left')
    
    # Display the full data with predictions
    st.write(data_with_predictions)
