import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('gb_model.pkl', 'rb'))

# Mapping for the target labels
label_mapping = {0: "Bad", 1: "Good"}

def predict_class(features):
    # Reshape the features array to match the expected input shape of the model
    features = np.array(features).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Map the predicted label to the corresponding class
    predicted_class = label_mapping[prediction[0]]

    return predicted_class

def main():
    st.title("Credit Scoring App")
    st.write("Please enter the following information to predict the credit score.")

    # Collect user input for each feature
    duration = st.number_input("Duration of Credit (in months)", min_value=0)
    credit_amount = st.number_input("Credit Amount", min_value=0)
    age = st.number_input("Age", min_value=0)
    existing_credits = st.number_input("Number of Existing Credits", min_value=0)

    # Make prediction when the user clicks the 'Predict' button
    if st.button("Predict"):
        features = [duration, credit_amount, age, existing_credits]
        prediction = predict_class(features)
        st.write("Predicted Credit Score:", prediction)

if __name__ == "__main__":
    main()
