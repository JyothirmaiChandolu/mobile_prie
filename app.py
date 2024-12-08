import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# Title and Subheader
st.title("Mobile Price Prediction")
st.subheader("Using LightGBM Machine Learning Model")

# Load Dataset
data = pd.read_excel('Mobile-Price-Prediction-cleaned_data.xlsx')
st.dataframe(data.head)

# Features and Target
X = data.drop(columns='Price')
y = data['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Evaluation Metrics
st.write("### Model Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R2): {r2:.2f}")

# User Inputs
st.write("### Enter Mobile Specifications:")
ratings = st.slider("Ratings out of 5", min_value=1.0, max_value=5.0, step=0.1)
ram = st.slider("RAM (GB)", min_value=1, max_value=16, value=4, step=1)
rom = st.slider("Internal Storage (GB)", min_value=8, max_value=512, value=64, step=8)
screen_size = st.slider("Screen Size (inches)", min_value=4.0, max_value=7.0, value=6.0, step=0.1)
primary_camera = st.slider("Primary Camera (MP)", min_value=8, max_value=108, value=48, step=1)
selfie_camera = st.slider("Selfie Camera (MP)", min_value=5, max_value=50, value=16, step=1)
battery = st.slider("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4000, step=100)

# Prediction
features = [ratings, ram, rom, screen_size, primary_camera, selfie_camera, battery]
scaled_features = scaler.transform([features])

if st.button("Predict Price"):
    predicted_price = model.predict(scaled_features)[0]
    st.success(f"The predicted price of the mobile is ₹{predicted_price:.2f}")
