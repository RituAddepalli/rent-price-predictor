from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from flask_cors import CORS  # ✅ add CORS

app = Flask(__name__)
CORS(app)  # ✅ allow frontend requests

MODEL_FILE = 'rent_model.pkl'
DATA_FILE = 'data/rent_data.csv'

# -----------------------------
# 1️⃣ Load or train model
# -----------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print("✅ Model loaded successfully!")
else:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    X = df[['area', 'bedrooms', 'age', 'location']]
    y = df['rent']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=300, random_state=42))  # more trees for stability
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    print("✅ Model trained and saved!")

# -----------------------------
# 2️⃣ Prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict_rent():
    data = request.get_json()
    try:
        input_df = pd.DataFrame([{
            'area': data.get('area', 0),
            'bedrooms': data.get('bedrooms', 0),
            'age': data.get('age', 0),
            'location': data.get('location', '')
        }])
    except Exception as e:
        return jsonify({'error': f'Invalid input: {e}'})

    try:
        # Base prediction from model
        prediction = model.predict(input_df)
        rent_value = float(prediction[0])

        # -----------------------------
        # Location multiplier
        # -----------------------------
        loc = data.get('location', '').lower()
        if loc == 'suburbs':
            rent_value *= 0.85  # suburbs cheaper
        elif loc == 'countryside':
            rent_value *= 0.65  # cheapest
        elif loc == 'near ocean':
            rent_value *= 1.1  # expensive
        # Downtown remains base

        # -----------------------------
        # Age depreciation
        # -----------------------------
        age = data.get('age', 0)
        rent_value *= max(0.2, 1 - age * 0.025)  # 2.5% reduction per year, min 20%

        # -----------------------------
        # Bedroom adjustment
        # -----------------------------
        bedrooms = data.get('bedrooms', 1)
        if bedrooms > 3:
            rent_value *= 1 + (bedrooms - 3) * 0.1  # 10% more per extra bedroom

        # Ensure minimum rent
        rent_value = max(5000, rent_value)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'})

    return jsonify({'predicted_rent': round(rent_value, 2)})

if __name__ == '__main__':
    app.run(debug=True)




