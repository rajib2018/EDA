import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Generic EDA Tool')

uploaded_file = st.file_uploader('Upload your data (CSV)')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['dataframe'] = df
    st.write("Data loaded successfully!")

if 'dataframe' in st.session_state:
    st.write("Displaying the uploaded data:")
    st.dataframe(st.session_state['dataframe'])
#else:
    #st.write("Please upload a data file to display.")

if 'dataframe' in st.session_state:
    st.write("Summary Statistics:")
    st.write(st.session_state['dataframe'].describe())
#else:
    #st.write("Please upload a data file to see summary statistics.")

if 'dataframe' in st.session_state:
    st.write("Missing Values:")
    st.write(st.session_state['dataframe'].isnull().sum())
#else:
    #st.write("Please upload a data file to see missing values.")

if 'dataframe' in st.session_state:
    df = st.session_state['dataframe']
    st.write("### Data Visualization")


    visualization_type = st.selectbox(
        'Select visualization type:',
        ['Histogram', 'Box Plot', 'Scatter Plot']
    )

    if visualization_type == 'Histogram':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for histogram:', numerical_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], ax=ax, kde=True)
            st.pyplot(fig)
        else:
            st.write("No numerical columns found for histogram.")

    elif visualization_type == 'Box Plot':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for box plot:', numerical_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_column], ax=ax)
            st.pyplot(fig)
        else:
            st.write("No numerical columns found for box plot.")

    elif visualization_type == 'Scatter Plot':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_column = st.selectbox('Select a numerical column for x-axis:', numerical_cols)
            y_column = st.selectbox('Select a numerical column for y-axis:', numerical_cols)
            if x_column != y_column:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax)
                st.pyplot(fig)
            else:
                st.write("Please select different columns for x and y axes.")
        else:
            st.write("Need at least two numerical columns for a scatter plot.")

else:
    st.write("Please upload a data file to visualize.")
