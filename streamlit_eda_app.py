import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Dubai Residential Data Explorer', layout='wide')
st.title('Dubai Residential Real Estate — Data Explorer')

@st.cache_data
def load_data(path='dubai_residential_classified1(1).csv'):
    df = pd.read_csv(path)
    if 'INSTANCE_DATE' in df.columns:
        df['INSTANCE_DATE'] = pd.to_datetime(df['INSTANCE_DATE'], errors='coerce')
    for col in ['ROOMS_EN','TRANS_VALUE','ACTUAL_AREA','price_per_sqm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.drop_duplicates().copy()

df = load_data()

st.sidebar.header('Filters')
filtered = df.copy()
if 'AREA_EN' in df.columns:
    areas = st.sidebar.multiselect('Area', sorted(df['AREA_EN'].dropna().unique()), default=[])
    if areas: filtered = filtered[filtered['AREA_EN'].isin(areas)]
if 'PROP_SB_TYPE_EN' in df.columns:
    types = st.sidebar.multiselect('Property Type', sorted(df['PROP_SB_TYPE_EN'].dropna().unique()), default=[])
    if types: filtered = filtered[filtered['PROP_SB_TYPE_EN'].isin(types)]
if 'value_band' in df.columns:
    bands = st.sidebar.multiselect('Value Band', sorted(df['value_band'].dropna().unique()), default=[])
    if bands: filtered = filtered[filtered['value_band'].isin(bands)]

c1,c2,c3,c4 = st.columns(4)
c1.metric('Rows', f'{len(filtered):,}')
c2.metric('Transaction Value Mean', f'{filtered["TRANS_VALUE"].mean():,.0f}' if 'TRANS_VALUE' in filtered else 'N/A')
c3.metric('Median Area', f'{filtered["ACTUAL_AREA"].median():,.2f}' if 'ACTUAL_AREA' in filtered else 'N/A')
c4.metric('Median Price/sqm', f'{filtered["price_per_sqm"].median():,.0f}' if 'price_per_sqm' in filtered else 'N/A')

st.subheader('Data Preview')
st.dataframe(filtered, use_container_width=True)

st.subheader('Distributions')
num = [c for c in ['TRANS_VALUE','ACTUAL_AREA','price_per_sqm','ROOMS_EN'] if c in filtered.columns]
for col in num:
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(filtered[col].dropna(), kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
    st.pyplot(fig)
    plt.close(fig)

st.subheader('Bivariate Analysis')
if 'ACTUAL_AREA' in filtered and 'TRANS_VALUE' in filtered:
    fig, ax = plt.subplots(figsize=(10,5))
    sample = filtered.sample(min(5000,len(filtered)), random_state=42)
    sns.scatterplot(data=sample, x='ACTUAL_AREA', y='TRANS_VALUE', alpha=.5, ax=ax)
    st.pyplot(fig); plt.close(fig)

st.subheader('Correlation Matrix')
if len(num) >= 2:
    fig, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(filtered[num].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig); plt.close(fig)

st.download_button('Download Filtered CSV', filtered.to_csv(index=False).encode('utf-8'), 'filtered_dubai_residential.csv', 'text/csv')
