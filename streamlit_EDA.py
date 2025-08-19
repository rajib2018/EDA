import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

st.title('Generic EDA Tool')

uploaded_file = st.file_uploader('Upload your data (CSV)')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['dataframe'] = df
    st.write("Data loaded successfully!")

if 'dataframe' in st.session_state:
    st.write("Displaying the uploaded data:")
    st.dataframe(st.session_state['dataframe'])
else:
    st.write("Please upload a data file to display.")

if 'dataframe' in st.session_state:
    st.write("Missing Values:")
    st.write(st.session_state['dataframe'].isnull().sum())
else:
    st.write("Please upload a data file to see missing values.")
