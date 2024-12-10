import streamlit as st
import pandas as pd
import pickle

st.header('House price predictor')
st.write('Please enter information about the house you seek to value')
above_ground_area = st.number_input("Above-ground area")
lot_area = st.slider('Lot area', min_value=0, max_value=1000)
overall_quality = st.selectbox('Overall quality', ('Above_Average', 'Average', 'Good', 'Very_Good', 'Excellent', 'Below_Average', 'Fair', 'Poor', 'Very_Excellent', 'Very_Poor'))
sale_condition = st.selectbox('Sale condition', ('Normal', 'Partial', 'Family', 'Abnorml', 'Alloca', 'AdjLand'))

X = pd.DataFrame({'Gr_Liv_Area': above_ground_area,
                  'Overall_Qual': overall_quality,
                  'Sale_Condition':sale_condition,
                  'Lot_Area': lot_area}, index = [0])

st.dataframe(X)

with open('lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

preds = model.predict(X)
st.write('Value: ', preds)