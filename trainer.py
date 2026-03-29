import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_temperature_model(file_path):
    # Step 1: Load the dataset
    try:
        print("Loading dataset...")
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: The file was not found. Please verify the file path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # Step 2: Preprocess the data
    try:
        print("Preprocessing data...")
        
        # Drop columns that are categorical, duplicates, or not useful for numeric prediction
        columns_to_drop = ['Country', 'Vlookup value', 'Year (copy)', 'Number of Records']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns_to_drop)
        
        # Remove rows containing missing values to prevent training errors
        df = df.dropna()

        # Check if the target column exists
        if 'Temperature' not in df.columns:
            print("Error: Target column 'Temperature' not found in the dataset.")
            return

        # Separate features (X) and the target variable (y)
        X = df.drop(columns=['Temperature'])
        y = df['Temperature']
        
        # Convert any remaining categorical text columns into numerical format
        X = pd.get_dummies(X, drop_first=True)

    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        return

    # Step 3: Split the data into training and testing sets
    try:
        print("Splitting data...")
        # 80% of data used for training, 20% used for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"An error occurred during data splitting: {e}")
        return

    # Step 4: Initialize and train the machine learning model
    try:
        print("Training the model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return

    # Step 5: Make predictions and evaluate performance
    try:
        print("Evaluating the model...")
        predictions = model.predict(X_test)
        
        # Calculate error metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("Model training complete.")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Return the trained model for further use if needed
        return model
        
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        return None

# Entry point of the script
if __name__ == "__main__":
    # Define the dataset file name
    dataset_file = r"D:\vidyarthi_ut\Global_Warming.csv"
    
    # Run the function
    trained_model = train_temperature_model(dataset_file)
# Save the model if training was successful
    if trained_model is not None:
        # This saves the model to a file named 'temperature_model.pkl'
        joblib.dump(trained_model, 'temperature_model.pkl')
        print("Model saved successfully as 'temperature_model.pkl'!")