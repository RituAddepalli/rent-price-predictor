import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_FILE = 'rent_model.pkl'
DATA_FILE = 'data/rent_data.csv'  # make sure your CSV is in the data folder

# -----------------------------
# 1️⃣ Load or train model
# -----------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print("✅ Loaded existing model.")
else:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find dataset at {DATA_FILE}")

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
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    print("✅ Model trained and saved on the spot!")

# -----------------------------
# 2️⃣ User input
# -----------------------------
user_input = {
    'area': 2500,
    'bedrooms': 1,
    'age': 10,
    'location': 'Downtown'
}

input_df = pd.DataFrame([user_input])

# -----------------------------
# 3️⃣ Predict safely
# -----------------------------
try:
    predicted_rent = model.predict(input_df)
    rent_value = float(predicted_rent[0])

    # Apply scaling
    scaling_factor = 0.4
    rent_value = rent_value * scaling_factor

    # Enforce minimum rent
    rent_value = max(rent_value, 10000)

    print(f"Predicted monthly rent: ₹{rent_value:.2f}")
except Exception as e:
    print(f"⚠ Prediction failed: {e}")


