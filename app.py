
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("cars.csv")
X = df[['year', 'odometer']]
y = df['price']
model = LinearRegression().fit(X, y)

st.title("🚗 Car Price Predictor")
st.markdown("Enter car details to predict its price.")

year = st.slider("Year of Manufacture", 1990, 2025, 2015)
odometer = st.number_input("Odometer (miles)", min_value=0, max_value=300000, value=60000)

input_data = pd.DataFrame([[year, odometer]], columns=['year', 'odometer'])
prediction = model.predict(input_data)

st.subheader("📈 Predicted Price:")
st.write(f"${prediction[0]:,.2f}")

if st.checkbox("Show Dataset Summary"):
    st.write(df.describe())
