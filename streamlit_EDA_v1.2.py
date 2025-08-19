import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import bokeh.plotting as bk_plt
import altair as alt
import hvplot.pandas
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.plotting import figure
from bokeh.models import Whisker


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
    st.write("Summary Statistics:")
    st.write(st.session_state['dataframe'].describe())
else:
    st.write("Please upload a data file to see summary statistics.")

if 'dataframe' in st.session_state:
    st.write("Missing Values:")
    st.write(st.session_state['dataframe'].isnull().sum())
else:
    st.write("Please upload a data file to see missing values.")

if 'dataframe' in st.session_state:
    df = st.session_state['dataframe']
    st.write("### Data Visualization")

    # Visualization Recommendations
    st.write("#### Visualization Recommendations:")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include=np.datetime64).columns.tolist()

    if numerical_cols:
        st.write(f"- **Numerical Data:**")
        st.write(f"  - **Histograms/Box Plots:** Use for visualizing the distribution of a single numerical column. Matplotlib/Seaborn are good for static plots. Plotly and Bokeh provide interactive plots. Altair uses a declarative syntax which can be powerful. hvPlot offers a concise way to create interactive plots.")
        st.write(f"  - **Scatter Plots:** Use for visualizing the relationship between two numerical columns. Plotly and Bokeh offer interactivity (zooming, panning, tooltips). Altair's declarative approach is useful for complex mappings. hvPlot is concise for quick interactive plots.")
    if categorical_cols:
        st.write(f"- **Categorical Data:**")
        st.write(f"  - **Bar Plots/Count Plots:** Use to visualize the distribution of categories. Most libraries support this.")
    if date_cols:
        st.write(f"- **Date Data:**")
        st.write(f"  - **Time Series Plots:** Use to visualize trends over time. Plotly, Altair, and hvPlot are well-suited for interactive time series visualizations.")
    if numerical_cols and categorical_cols:
         st.write(f"- **Numerical and Categorical Data:**")
         st.write(f"  - **Box Plots/Violin Plots:** Use to compare numerical distributions across categories. Interactive libraries like Plotly and Bokeh can enhance exploration.")

    visualization_type = st.selectbox(
        'Select visualization type:',
        ['Histogram (Matplotlib/Seaborn)', 'Box Plot (Matplotlib/Seaborn)', 'Scatter Plot (Matplotlib/Seaborn)',
         'Histogram (Plotly)', 'Box Plot (Plotly)', 'Scatter Plot (Plotly)',
         'Histogram (Bokeh)', 'Box Plot (Bokeh)', 'Scatter Plot (Bokeh)',
         'Histogram (Altair)', 'Box Plot (Altair)', 'Scatter Plot (Altair)',
         'Histogram (hvPlot)', 'Box Plot (hvPlot)', 'Scatter Plot (hvPlot)']
    )

    if visualization_type == 'Histogram (Matplotlib/Seaborn)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for histogram:', numerical_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], ax=ax, kde=True)
            st.pyplot(fig)
        else:
            st.write("No numerical columns found for histogram.")

    elif visualization_type == 'Box Plot (Matplotlib/Seaborn)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for box plot:', numerical_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_column], ax=ax)
            st.pyplot(fig)
        else:
            st.write("No numerical columns found for box plot.")

    elif visualization_type == 'Scatter Plot (Matplotlib/Seaborn)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_column = st.selectbox('Select a numerical column for x-axis:', numerical_cols, key='scatter_mpl_x')
            y_column = st.selectbox('Select a numerical column for y-axis:', numerical_cols, key='scatter_mpl_y')
            if x_column and y_column and x_column != y_column:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax)
                st.pyplot(fig)
            else:
                st.write("Please select different numerical columns for x and y axes.")
        else:
            st.write("Need at least two numerical columns for a scatter plot.")

    elif visualization_type == 'Histogram (Plotly)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for histogram:', numerical_cols, key='hist_plotly')
            fig = px.histogram(df, x=selected_column)
            st.plotly_chart(fig)
        else:
            st.write("No numerical columns found for histogram.")

    elif visualization_type == 'Box Plot (Plotly)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for box plot:', numerical_cols, key='box_plotly')
            fig = px.box(df, y=selected_column)
            st.plotly_chart(fig)
        else:
            st.write("No numerical columns found for box plot.")

    elif visualization_type == 'Scatter Plot (Plotly)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_column = st.selectbox('Select a numerical column for x-axis:', numerical_cols, key='scatter_plotly_x')
            y_column = st.selectbox('Select a numerical column for y-axis:', numerical_cols, key='scatter_plotly_y')
            if x_column and y_column and x_column != y_column:
                fig = px.scatter(df, x=x_column, y=y_column)
                st.plotly_chart(fig)
            else:
                st.write("Please select different numerical columns for x and y axes.")
        else:
            st.write("Need at least two numerical columns for a scatter plot.")

    elif visualization_type == 'Histogram (Bokeh)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for histogram:', numerical_cols, key='hist_bokeh')
            hist, edges = np.histogram(df[selected_column].dropna(), bins=20)
            p = bk_plt.figure(title="Histogram", x_axis_label=selected_column, y_axis_label="Frequency")
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
            st.bokeh_chart(p)
        else:
            st.write("No numerical columns found for histogram.")

    elif visualization_type == 'Box Plot (Bokeh)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for box plot:', numerical_cols, key='box_bokeh')
            # Bokeh box plot requires some data preparation


            data = pd.DataFrame(dict(score=df[selected_column].dropna()))
            quantiles = data.quantile([0.25, 0.5, 0.75]).to_dict()['score']
            iqr = quantiles[0.75] - quantiles[0.25]
            upper_bound = quantiles[0.75] + 1.5 * iqr
            lower_bound = quantiles[0.25] - 1.5 * iqr

            outliers = data[(data['score'] > upper_bound) | (data['score'] < lower_bound)]
            stem_top = min(upper_bound, data['score'].max())
            stem_bottom = max(lower_bound, data['score'].min())

            source = ColumnDataSource(dict(
                score=[quantiles[0.25], quantiles[0.5], quantiles[0.75]],
                x=[''] * 3
            ))
            outlier_source = ColumnDataSource(dict(score=outliers['score'].tolist(), x=[''] * len(outliers)))

            p = figure(title="Box Plot", x_range=[''], y_axis_label=selected_column)
            p.vbar(x=[''], top=quantiles[0.75], bottom=quantiles[0.25], width=0.2, source=source, line_color="black")
            p.segment(x0=[''], x1=[''], y0=stem_bottom, y1=quantiles[0.25], line_color="black")
            p.segment(x0=[''], x1=[''], y0=quantiles[0.75], y1=stem_top, line_color="black")
            p.rect(x=[''], y=quantiles[0.5], width=0.3, height=0.01, line_color="black", fill_color=None)
            p.scatter(x='x', y='score', source=outlier_source, size=6, color='red', alpha=0.5)

            st.bokeh_chart(p)

        else:
            st.write("No numerical columns found for box plot.")

    elif visualization_type == 'Scatter Plot (Bokeh)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_column = st.selectbox('Select a numerical column for x-axis:', numerical_cols, key='scatter_bokeh_x')
            y_column = st.selectbox('Select a numerical column for y-axis:', numerical_cols, key='scatter_bokeh_y')
            if x_column and y_column and x_column != y_column:
                p = bk_plt.figure(title="Scatter Plot", x_axis_label=x_column, y_axis_label=y_column)
                p.circle(x=df[x_column], y=df[y_column], size=5)
                st.bokeh_chart(p)
            else:
                st.write("Please select different numerical columns for x and y axes.")
        else:
            st.write("Need at least two numerical columns for a scatter plot.")

    elif visualization_type == 'Histogram (Altair)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for histogram:', numerical_cols, key='hist_altair')
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(selected_column, bin=True),
                y='count()',
            ).properties(
                title=f'Histogram of {selected_column}'
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No numerical columns found for histogram.")

    elif visualization_type == 'Box Plot (Altair)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for box plot:', numerical_cols, key='box_altair')
            chart = alt.Chart(df).mark_boxplot().encode(
                y=alt.Y(selected_column),
            ).properties(
                title=f'Box Plot of {selected_column}'
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No numerical columns found for box plot.")

    elif visualization_type == 'Scatter Plot (Altair)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_column = st.selectbox('Select a numerical column for x-axis:', numerical_cols, key='scatter_altair_x')
            y_column = st.selectbox('Select a numerical column for y-axis:', numerical_cols, key='scatter_altair_y')
            if x_column and y_column and x_column != y_column:
                chart = alt.Chart(df).mark_point().encode(
                    x=x_column,
                    y=y_column
                ).properties(
                    title=f'Scatter Plot of {y_column} vs {x_column}'
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("Please select different numerical columns for x and y axes.")
        else:
            st.write("Need at least two numerical columns for a scatter plot.")

    elif visualization_type == 'Histogram (hvPlot)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for histogram:', numerical_cols, key='hist_hvplot')
            # hvPlot returns a HoloViews object
            plot = df[selected_column].hvplot.hist()
            st.bokeh_chart(hvplot.render(plot))
        else:
            st.write("No numerical columns found for histogram.")

    elif visualization_type == 'Box Plot (hvPlot)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            selected_column = st.selectbox('Select a numerical column for box plot:', numerical_cols, key='box_hvplot')
            plot = df.hvplot.box(y=selected_column)
            st.bokeh_chart(hvplot.render(plot))
        else:
            st.write("No numerical columns found for box plot.")

    elif visualization_type == 'Scatter Plot (hvPlot)':
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_column = st.selectbox('Select a numerical column for x-axis:', numerical_cols, key='scatter_hvplot_x')
            y_column = st.selectbox('Select a numerical column for y-axis:', numerical_cols, key='scatter_hvplot_y')
            if x_column and y_column and x_column != y_column:
                plot = df.hvplot.scatter(x=x_column, y=y_column)
                st.bokeh_chart(hvplot.render(plot))
            else:
                st.write("Please select different numerical columns for x and y axes.")
        else:
            st.write("Need at least two numerical columns for a scatter plot.")

else:
    st.write("Please upload a data file to visualize.")
