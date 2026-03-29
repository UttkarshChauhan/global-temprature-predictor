import gradio as gr
import pandas as pd
import joblib
import os

# Step 1: Load the trained model
# We set this up outside the prediction function so it only loads once
model_path = r"D:\vidyarthi_ut\temperature_model.pkl"
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

# Step 2: Define the prediction and error handling logic
def predict_temperature(year, anomaly, co2_emissions):
    # Error Handling: Check if model loaded successfully
    if model is None:
        return "System Error: 'temperature_model.pkl' not found. Please ensure the model is trained and in the same folder."
    
    # Error Handling: Validate that inputs are not empty
    if not str(year).strip() or not str(anomaly).strip() or not str(co2_emissions).strip():
        return "Input Error: All fields are required. Please fill in all values."

    # Error Handling: Validate that inputs are properly formatted numbers
    try:
        year_val = float(year)
        anomaly_val = float(anomaly)
        co2_val = float(co2_emissions)
    except ValueError:
        return "Input Error: Invalid characters detected. Please enter numerical values only."
        
    # Error Handling: Logical validation for the Year
    if year_val < 1800 or year_val > 2200:
        return "Input Error: Please enter a realistic year (between 1800 and 2200)."
        
    if co2_val < 0:
        return "Input Error: CO2 Emissions cannot be a negative number."

    # Step 3: Format the data exactly as the model expects it
    try:
        input_data = pd.DataFrame({
            'Year': [year_val],
            'Anomaly Standardised': [anomaly_val],
            'CO2 Emissions': [co2_val]
        })
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        return f"Predicted Temperature: {prediction:.4f} °C"
        
    except Exception as e:
        return f"Prediction Error: An unexpected error occurred during calculation - {e}"

# Step 4: Build the UI layout
interface = gr.Interface(
    fn=predict_temperature,
    inputs=[
        gr.Textbox(label="Year (e.g., 2025)", placeholder="Enter the year..."),
        gr.Textbox(label="Anomaly Standardised", placeholder="Enter the temperature anomaly..."),
        gr.Textbox(label="CO2 Emissions", placeholder="Enter total CO2 emissions...")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Global Temperature Predictor",
    description="Enter the climate data features below to predict the temperature. The system will catch invalid inputs automatically.",
    
)

# Entry point
if __name__ == "__main__":
    # Launch the web-based GUI
    print("Starting the interface...")
    interface.launch(share=True,theme="default")