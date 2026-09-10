# Darin Ashraf Aly

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(

    page_title="Air Quality Analysis",

    page_icon="🌍",

    layout="wide"

)

st.title(
    "Air Quality Analysis & Prediction"
)

st.markdown(
    """
    This application analyzes air quality data and predicts
    Air Quality Index (AQI) using machine learning.
    """
)

@st.cache_data
def load_data():

    return pd.read_csv(
        "models/cleaned_air_quality.csv"
    )

@st.cache_data
def load_results():

    return pd.read_csv(
        "models/model_results.csv"
    )


@st.cache_data
def load_evaluation():

    return pd.read_csv(
        "models/evaluation_data.csv"
    )

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_model.pkl"
    )

    return model

@st.cache_data
def load_features():

    return joblib.load(
        "models/model_features.pkl"
    )

@st.cache_data
def load_best_model_name():

    return joblib.load(
        "models/best_model_name.pkl"
    )

df = load_data()

results = load_results()

evaluation_data = load_evaluation()

best_model = load_model()

features = load_features()

best_model_name = load_best_model_name()


st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(

    "Go to",

    [

        "EDA Dashboard",

        "AQI Prediction",

        "Model Evaluation",

        "Dataset Information"

    ]

)


if page == "EDA Dashboard":

    st.header(
        "Air Quality EDA Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Average AQI",

            f"{df['aqi'].mean():.2f}"

        )


    with col2:

        if "pm2_5" in df.columns:

            st.metric(

                "Average PM2.5",

                f"{df['pm2_5'].mean():.2f}"

            )


    with col3:

        st.metric(

            "AQI Records",

            f"{len(df):,}"

        )


    with col4:

        if "co" in df.columns:

            st.metric(

                "Average CO",

                f"{df['co'].mean():.2f}"

            )


    st.divider()

    if "date" in df.columns:

        st.subheader(
            "Average AQI Over Time"
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )


        time_df = (
            df
            .dropna(subset=["date"])
            .groupby(df["date"].dt.year)["aqi"]
            .mean()
        )

        time_df.index = time_df.index.astype(str)

        st.line_chart(time_df)

    if "date" in df.columns:

        st.subheader(
            "Average AQI by Month"
        )


        month_df = (

            df

            .dropna(
                subset=["date"]
            )

            .groupby(
                df["date"].dt.month
            )["aqi"]

            .mean()

        )


        month_names = {

            1: "Jan",

            2: "Feb",

            3: "Mar",

            4: "Apr",

            5: "May",

            6: "Jun",

            7: "Jul",

            8: "Aug",

            9: "Sep",

            10: "Oct",

            11: "Nov",

            12: "Dec"

        }


        month_df.index = [

            month_names[x]

            for x in month_df.index

        ]

        st.line_chart(
            month_df
        )

    col1, col2 = st.columns(2)

    with col1:

        if "county" in df.columns:

            st.subheader(
                "Average AQI by County"
            )


            county_df = (

                df

                .groupby(
                    "county"
                )["aqi"]

                .mean()

                .sort_values(
                    ascending=False
                )

                .head(10)

            )

            st.bar_chart(
                county_df
            )

    with col2:

        if "sitename" in df.columns:

            st.subheader(
                "Top 10 Monitoring Stations"
            )


            station_df = (

                df

                .groupby(
                    "sitename"
                )["aqi"]

                .mean()

                .sort_values(
                    ascending=False
                )

                .head(10)

            )


            st.bar_chart(
                station_df
            )

elif page == "AQI Prediction":

    st.header(
        "AQI Prediction"
    )


    st.write(

        "Enter the environmental and pollutant "
        "measurements to predict the AQI."

    )

    input_values = {}


    cols = st.columns(3)


    for i, feature in enumerate(features):

        with cols[i % 3]:

            min_value = float(
                df[feature].min()
            )


            max_value = float(
                df[feature].max()
            )


            median_value = float(
                df[feature].median()
            )


            input_values[feature] = st.number_input(

                feature.upper(),

                min_value=min_value,

                max_value=max_value,

                value=median_value

            )


    st.divider()

    if st.button(

        "Predict AQI",

        use_container_width=True

    ):


        input_df = pd.DataFrame(
            [input_values]
        )



        input_df = input_df[
            features
        ]



        prediction_value = best_model.predict(
            input_df
        )[0]

        st.success(
            "Prediction completed successfully!"
        )


        st.metric(

            "Predicted AQI",

            f"{prediction_value:.2f}"

        )

        if prediction_value <= 50:

            category = "Good"

        elif prediction_value <= 100:

            category = "Moderate"

        elif prediction_value <= 150:

            category = (
                "Unhealthy for Sensitive Groups"
            )

        elif prediction_value <= 200:

            category = "Unhealthy"

        elif prediction_value <= 300:

            category = "Very Unhealthy"

        else:

            category = "Hazardous"


        st.info(

            f"AQI Category: **{category}**"

        )


        st.write(

            f"Model used: **{best_model_name}**"

        )

elif page == "Model Evaluation":

    st.header(
        "Model Evaluation"
    )

    st.subheader(
        "Model Comparison"
    )


    st.dataframe(

        results,

        use_container_width=True,

        hide_index=True

    )


    st.success(

        f"Best Model: {best_model_name}"

    )

    best_row = results[

        results["Model"] == best_model_name

    ].iloc[0]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "MAE",

            f"{best_row['MAE']:.4f}"

        )


    with col2:

        st.metric(

            "RMSE",

            f"{best_row['RMSE']:.4f}"

        )


    with col3:

        st.metric(

            "R² Score",

            f"{best_row['R2 Score']:.4f}"

        )


    st.divider()

    st.subheader(
        "Actual vs Predicted AQI"
    )


    plot_df = evaluation_data.copy()


    plot_df = plot_df.sort_values(
        "actual"
    )

    if len(plot_df) > 1000:

        indices = np.linspace(

            0,

            len(plot_df) - 1,

            1000

        ).astype(int)


        plot_df = plot_df.iloc[
            indices
        ]


    fig, ax = plt.subplots(

        figsize=(10, 5)

    )


    ax.plot(

        plot_df["actual"].values,

        label="Actual AQI"

    )


    ax.plot(

        plot_df["predicted"].values,

        label="Predicted AQI"

    )


    ax.set_xlabel(
        "Observations"
    )


    ax.set_ylabel(
        "AQI"
    )


    ax.set_title(
        "Actual vs Predicted AQI"
    )


    ax.legend()


    st.pyplot(fig)


    plt.close(fig)

    st.subheader(
        "Prediction Error by AQI Range"
    )


    eval_df = evaluation_data.copy()


    eval_df["error"] = abs(

        eval_df["actual"]

        -

        eval_df["predicted"]

    )


    bins = [

        0,

        25,

        50,

        75,

        100,

        127

    ]


    labels = [

        "0-25",

        "26-50",

        "51-75",

        "76-100",

        "101-127"

    ]


    eval_df["aqi_range"] = pd.cut(

        eval_df["actual"],

        bins=bins,

        labels=labels,

        include_lowest=True

    )


    mae_by_range = (

        eval_df

        .groupby(

            "aqi_range",

            observed=False

        )["error"]

        .mean()

    )


    fig, ax = plt.subplots(

        figsize=(8, 5)

    )


    ax.bar(

        mae_by_range.index.astype(str),

        mae_by_range.values

    )


    ax.set_xlabel(
        "Actual AQI Range"
    )


    ax.set_ylabel(
        "Mean Absolute Error"
    )


    ax.set_title(
        "Prediction Error by AQI Range"
    )


    st.pyplot(fig)


    plt.close(fig)

    st.subheader(
        "Distribution of Prediction Errors"
    )


    residuals = (

        evaluation_data["predicted"]

        -

        evaluation_data["actual"]

    )


    fig, ax = plt.subplots(

        figsize=(8, 5)

    )


    ax.hist(

        residuals,

        bins=50

    )


    ax.axvline(

        0,

        linestyle="--"

    )


    ax.set_xlabel(
        "Residual"
    )


    ax.set_ylabel(
        "Frequency"
    )


    ax.set_title(
        "Distribution of Prediction Errors"
    )


    st.pyplot(fig)


    plt.close(fig)

elif page == "Dataset Information":

    st.header(
        "Dataset Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Rows",

            f"{df.shape[0]:,}"

        )


    with col2:

        st.metric(

            "Columns",

            df.shape[1]

        )


    with col3:

        st.metric(

            "Duplicates",

            df.duplicated().sum()

        )

    st.divider()

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(

        df.head(100),

        use_container_width=True

    )


    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(

        df.describe(),

        use_container_width=True

    )