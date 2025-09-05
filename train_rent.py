import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os

DATA_FILE = 'data/rent_data.csv'
MODEL_FILE = 'rent_model.pkl'

# Load data
df = pd.read_csv(DATA_FILE)
X = df[['area', 'bedrooms', 'age', 'location']]
y = df['rent']

# Preprocessor for location
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])],
    remainder='passthrough'
)

# Pipeline with Random Forest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_FILE)
print("✅ Model trained and saved!")

