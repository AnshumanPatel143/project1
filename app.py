from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

app = Flask(__name__)
CORS(app)

DATASET_PATH = r'C:\Users\Anshu2006\OneDrive\Desktop\DATA DCIENCE PROJECT\House1_Price_Multiplied.csv'

def load_and_preprocess_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    # Encode categorical variables
    le_state = LabelEncoder()
    le_city = LabelEncoder()
    le_condition = LabelEncoder()
    le_garage = LabelEncoder()

    df['State'] = le_state.fit_transform(df['State'])
    df['Metro City'] = le_city.fit_transform(df['Metro City'])
    df['Condition'] = le_condition.fit_transform(df['Condition'])
    df['Garage'] = le_garage.fit_transform(df['Garage'])

    # Filter out outliers
    df = df[df['Price'] < df['Price'].quantile(0.99)]

    # Create new feature
    df['Bedrooms_Bathrooms'] = df['Bedrooms'] * df['Bathrooms']

    # Features and target
    X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'State', 'Metro City', 'Condition', 'Garage', 'Bedrooms_Bathrooms']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model Evaluation Metrics:")
    print(f"R² Score  : {r2:.4f}")
    print(f"MAE       : {mae:.2f}")
    print(f"RMSE      : {rmse:.2f}")

    return model, le_state, le_city, le_condition, le_garage, df, r2, mae, rmse

# Load model and encoders
try:
    model, le_state, le_city, le_condition, le_garage, df, r2_value, mae_value, rmse_value = load_and_preprocess_data()
    print("Model and data loaded successfully")
except Exception as e:
    print(f"Error initializing backend: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['state', 'city', 'area', 'bedrooms', 'bathrooms', 'floors', 'condition', 'garage']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        state = data['state']
        city = data['city']
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        floors = int(data['floors'])
        condition = data['condition']
        garage = data['garage']

        if not (500 <= area <= 5000):
            return jsonify({"error": "Area must be between 500 and 5000 sq ft"}), 400

        if state not in le_state.classes_ or city not in le_city.classes_ \
           or condition not in le_condition.classes_ or garage not in le_garage.classes_:
            return jsonify({"error": "Invalid categorical input"}), 400

        input_data = {
            'State': le_state.transform([state])[0],
            'Metro City': le_city.transform([city])[0],
            'Condition': le_condition.transform([condition])[0],
            'Garage': le_garage.transform([garage])[0],
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Floors': floors,
            'Bedrooms_Bathrooms': bedrooms * bathrooms
        }

        input_df = pd.DataFrame([input_data], columns=[
            'Area', 'Bedrooms', 'Bathrooms', 'Floors',
            'State', 'Metro City', 'Condition', 'Garage', 'Bedrooms_Bathrooms'
        ])

        predicted_price = float(max(0, model.predict(input_df)[0]))

        city_encoded = le_city.transform([city])[0]
        city_data = df[df['Metro City'] == city_encoded]
        avg_price = float(max(0, city_data['Price'].mean()))
        min_price = float(max(0, city_data['Price'].min()))
        max_price_val = float(max(0, city_data['Price'].max()))

        return jsonify({
            "predicted_price": predicted_price,
            "avg_price": avg_price,
            "min_price": min_price,
            "max_price": max_price_val
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        "r2_score": round(r2_value, 4),
        "mae": round(mae_value, 2),
        "rmse": round(rmse_value, 2)
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
