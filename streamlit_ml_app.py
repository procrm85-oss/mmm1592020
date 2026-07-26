import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Dubai Real Estate ML Predictor', layout='wide')
st.title('Dubai Residential Real Estate — ML Prediction')

bundle = joblib.load('dubai_real_estate_best_model.joblib')
model = bundle['model']
features = bundle['features']

st.sidebar.header('Property Inputs')
row = {}
for c in features:
    if c in ['AREA_EN','PROP_SB_TYPE_EN','size_category','value_band']:
        row[c] = st.sidebar.text_input(c, '')
    elif c == 'ACTUAL_AREA':
        row[c] = st.sidebar.number_input(c, min_value=0.0, value=100.0)
    elif c == 'ROOMS_EN':
        row[c] = st.sidebar.number_input(c, min_value=0.0, value=2.0, step=1.0)
    elif c == 'price_per_sqm':
        row[c] = st.sidebar.number_input(c, min_value=0.0, value=15000.0)
    elif c == 'YEAR':
        row[c] = st.sidebar.number_input(c, min_value=2000, max_value=2100, value=2026, step=1)
    elif c == 'MONTH':
        row[c] = st.sidebar.number_input(c, min_value=1, max_value=12, value=1, step=1)
    elif c == 'DAY':
        row[c] = st.sidebar.number_input(c, min_value=1, max_value=31, value=1, step=1)

input_df = pd.DataFrame([row], columns=features)
st.subheader('Input Data')
st.dataframe(input_df, use_container_width=True)

if st.button('Predict Transaction Value'):
    prediction = model.predict(input_df)[0]
    st.success(f'Estimated Transaction Value: {prediction:,.2f}')

st.info('The prediction is based on the selected model trained on the uploaded Dubai residential dataset.')
