#  Global Temperature Predictor

##  Overview
The **Global Temperature Predictor** is a machine learning tool that forecasts a country's average annual temperature based on its historical climate data and greenhouse gas footprint. 

Powered by a **Random Forest Regressor**, the model uses three key inputs—**Year**, **Country**, and **CO2 Emissions**—to accurately predict local temperature shifts. Designed with a robust data-cleaning pipeline and error handling, this tool empowers researchers and policymakers to conduct scenario testing and visualize the local climate impacts of varying carbon policies.

---

##  Features
- **Robust Error Handling**: Gracefully handles missing files, empty datasets, and parsing errors.
- **Automated Data Cleaning**: Automatically filters out missing values and tracks data loss.
- **Categorical Encoding**: Converts text-based country names into numerical formats suitable for machine learning.
- **High Accuracy**: Utilizes a Random Forest Regressor to capture complex, non-linear relationships between emissions and temperature.
- **Evaluation Metrics**: Automatically calculates and outputs Mean Squared Error (MSE) and R-squared ($R^2$) scores.

---

##  Prerequisites
To run this project, you need Python installed on your machine along with the following libraries:

- `pandas`
- `scikit-learn`

You can install the required dependencies using pip:
``bash
pip install pandas scikit-learn
 Dataset Requirements
The script expects a CSV file named Global Warming.csv in the same directory. The dataset must contain at least the following columns:

Country: The name or ISO code of the country.

Year: The year of the record.

CO2 Emissions: The recorded CO2 emissions for that year.

Temperature: The recorded average temperature (Target Variable).

 Usage
Clone the repository (or download the script).

Add your dataset: Ensure Global Warming.csv is in the root folder next to the script.

Run the script:

Bash
python model_trainer.py
(Note: Replace model_trainer.py with whatever you named your Python file).

Expected Output
When run successfully, the terminal will display the pipeline's progress and the model's evaluation metrics:

Plaintext
 Successfully loaded the dataset.
 Data preprocessing complete.
 Training the Random Forest Regressor...

--- Model Evaluation ---
Mean Squared Error (MSE): 0.5309
R-squared (R2) Score:     0.9925
------------------------

 Pipeline executed successfully.
 How it Works
Load Data: The script safely ingests the CSV file, catching any file-not-found or parsing errors.

Preprocess Data: It drops rows with missing CO2 or Temperature values and uses LabelEncoder to convert the Country column into numeric identifiers.

Train Model: The data is split into an 80% training set and a 20% testing set. A RandomForestRegressor (with 100 estimators) is trained on the features (Year, CO2 Emissions, Country_Encoded).

Evaluate: The model predicts temperatures for the test set and calculates its accuracy.

📝 License
