import streamlit as st 
import numpy as np
import pandas as pd
import pickle as pkl
## Let's load our model
def load_model():
    with open('modelh5.pkl', 'rb') as file:
        model = pkl.load(file)
    return model

def main():
    st.set_page_config(page_title="🏠 House Price Predictor 🏡", layout="centered")

    st.title("House Price Prediction App🏠🏚")
    st.write("Predict house prices using Linear Regression")
    st.markdown("---")

    model = load_model()
    
    area = st.number_input("Area (in sq ft)", min_value=500, max_value=5000, step=100)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=5, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=4, step=1)
    floors = st.number_input("Number of Floors", min_value=1, max_value=3, step=1)

    if st.button('💲💲Predict Price💸💸'):
        input_data = pd.DataFrame([[area, bedrooms, bathrooms, floors]],
                                  columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors'])
        prediction = model.predict(input_data)[0]
        st.success(f"🏡 Estimated House Price: ₹{prediction:,.2f}")
        
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit and scikit-learn")

if __name__ == "__main__":
    main()
