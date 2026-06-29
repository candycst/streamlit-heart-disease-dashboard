import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")
st.title("❤️ Heart Disease Prediction Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease_uci.csv")

    df['sex'] = df['sex'].astype('category')
    df['cp'] = df['cp'].astype('category')
    df['fbs'] = df['fbs'].astype('category')
    df['restecg'] = df['restecg'].astype('category')
    df['exang'] = df['exang'].astype('category')
    df['slope'] = df['slope'].astype('category')
    df['ca'] = df['ca'].astype('category')
    df['thal'] = df['thal'].astype('category')

    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    return df

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔎 Filters")

# INTERACTIVE FEATURE 1: Dropdown for Sex
sex_filter = st.sidebar.selectbox(
    "Select Sex",
    ["All"] + sorted(df["sex"].unique().tolist())
)

# INTERACTIVE FEATURE 2: Dropdown for Chest Pain Type
cp_filter = st.sidebar.selectbox(
    "Chest Pain Type",
    ["All"] + sorted(df["cp"].unique().tolist())
)

# INTERACTIVE FEATURE 3: Slider for Age Range
age_range = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)

# Additional filter for predictive model: Search Box for Exact Age
search_age = st.sidebar.number_input(
    "Search Exact Age for Prediction",
    min_value=int(df["age"].min()),
    max_value=int(df["age"].max()),
    value=int(df["age"].mean())
)

# =========================
# APPLY FILTERS  ← moved up before any use of filtered_df
# =========================
filtered_df = df.copy()

if sex_filter != "All":
    filtered_df = filtered_df[filtered_df["sex"] == sex_filter]

if cp_filter != "All":
    filtered_df = filtered_df[filtered_df["cp"] == cp_filter]

filtered_df = filtered_df[
    (filtered_df["age"] >= age_range[0]) &
    (filtered_df["age"] <= age_range[1])
]

# Filter for the search_age to use in the predictive model
person_for_prediction = df[df["age"] == search_age].iloc[0] if not df[df["age"] == search_age].empty else None

# =========================
# DATA QUALITY MONITORING
# =========================
st.subheader("📊 Data Quality Monitoring")

missing_values = df.isnull().sum().sum()
missing_rate = (missing_values / df.size) * 100
duplicate_rows = df.duplicated().sum()

col1, col2, col3 = st.columns(3)
col1.metric("Missing Values", missing_values)
col2.metric("Missing Data Rate", f"{missing_rate:.2f}%")
col3.metric("Duplicate Records", duplicate_rows)

# =========================
# BUSINESS MONITORING  ← now safe to use filtered_df
# =========================
st.subheader("❤️ Business Monitoring")

disease_cases = filtered_df["target"].sum()
disease_rate = (disease_cases / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0

col1, col2 = st.columns(2)
col1.metric("Heart Disease Cases", int(disease_cases))
col2.metric("Disease Rate", f"{disease_rate:.1f}%")

# =========================
# SUMMARY METRICS
# =========================
st.subheader("📊 Summary Metrics")
col1, col2, col3 = st.columns(3)

if not filtered_df.empty:
    col1.metric("Total Records", len(filtered_df))
    col2.metric("Mean Age", f"{filtered_df['age'].mean():.2f}")
    col3.metric("Heart Disease Cases (Filtered)", int(filtered_df['target'].sum()))
else:
    col1.metric("Total Records", 0)
    col2.metric("Mean Age", "N/A")
    col3.metric("Heart Disease Cases (Filtered)", 0)

# =========================
# PREDICTIVE / ANALYTICAL OUTPUT
# =========================
st.subheader("🧠 Heart Disease Risk Prediction")

def calculate_risk_score(data_row):
    score = 0
    age = data_row.get("age", 0)
    chol = data_row.get("chol", 0)
    trestbps = data_row.get("trestbps", 0)
    thalach = data_row.get("thalach", 999)
    exang = data_row.get("exang", 0)
    cp = data_row.get("cp", -1)

    if age > 55: score += 1
    if chol > 240: score += 1
    if trestbps > 140: score += 1
    if thalach < 120: score += 1
    if exang == 1: score += 1
    if cp in [1, 2, 3]: score += 1
    return score

if person_for_prediction is not None:
    risk_score_value = calculate_risk_score(person_for_prediction)

    st.write(f"**Predicted Risk for Age {int(search_age)}:**")
    if risk_score_value <= 1:
        st.success(f"Low Risk (Score: {risk_score_value})")
    elif risk_score_value <= 3:
        st.warning(f"Medium Risk (Score: {risk_score_value})")
    else:
        st.error(f"High Risk (Score: {risk_score_value})")

    st.write("*(Note: This is a simplified, rule-based model for demonstration purposes only and should not be used for actual medical diagnosis.)*")
else:
    st.info(f"No data available for age {int(search_age)} to perform prediction.")

# =========================
# DATA PREVIEW
# =========================
st.subheader("📁 Filtered Data Preview")
if not filtered_df.empty:
    st.dataframe(filtered_df.head())
else:
    st.info("No data available for the selected filters.")

# =========================
# VISUALIZATIONS
# =========================
if filtered_df.empty:
    st.warning("No data available for the selected filters to display visualizations.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_df["age"], kde=True, ax=ax, color='skyblue')
        ax.set_title('Age Distribution (Filtered)')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x="sex", data=filtered_df, palette='viridis', ax=ax)
        ax.set_title('Gender Distribution (Filtered)')
        ax.set_xlabel('Sex (0 = Female, 1 = Male)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        st.subheader("Distribution of Cholesterol")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_df["chol"], kde=True, ax=ax, color='lightcoral')
        ax.set_title('Cholesterol Distribution (Filtered)')
        ax.set_xlabel('Cholesterol (mg/dl)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col2:
        st.subheader("Chest Pain Type")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x="cp", data=filtered_df, palette='coolwarm', ax=ax)
        ax.set_title('Distribution of Chest Pain Type (Filtered)')
        ax.set_xlabel('Chest Pain Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        st.subheader("Age vs Max Heart Rate")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x="age",
            y="thalach",
            hue="target",
            data=filtered_df,
            ax=ax,
            palette='coolwarm'
        )
        ax.set_title('Age vs. Maximum Heart Rate Achieved (Filtered)')
        ax.set_xlabel('Age')
        ax.set_ylabel('Maximum Heart Rate (thalach)')
        st.pyplot(fig)

        st.subheader("Distribution of Resting Blood Pressure")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_df["trestbps"], kde=True, ax=ax, color='mediumseagreen')
        ax.set_title('Resting Blood Pressure Distribution (Filtered)')
        ax.set_xlabel('Resting Blood Pressure (trestbps)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

# =========================
# HEATMAP
# =========================
if filtered_df.empty:
    st.warning("No data available for the selected filters to display the Correlation Heatmap.")
else:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    numerical_filtered_df = filtered_df.select_dtypes(include=np.number)
    sns.heatmap(numerical_filtered_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features (Filtered)')
    st.pyplot(fig)


# =========================
# DATA DRIFT ANALYSIS
# =========================

st.header("📉 Data Drift Analysis")

# Baseline vs Current comparison (Age as example feature)
baseline_mean_age = df["age"].mean()
current_mean_age = filtered_df["age"].mean()

age_drift = abs(current_mean_age - baseline_mean_age)

col1, col2, col3 = st.columns(3)

col1.metric("Baseline Mean Age", f"{baseline_mean_age:.2f}")
col2.metric("Current Mean Age", f"{current_mean_age:.2f}")
col3.metric("Age Drift", f"{age_drift:.2f}")

# Simple drift interpretation
if age_drift > 5:
    st.error("⚠️ Significant data drift detected in Age distribution")
elif age_drift > 2:
    st.warning("⚠️ Moderate data drift detected")
else:
    st.success("✅ No significant data drift detected")


features = ["age", "chol", "trestbps"]

for f in features:
    st.write(f"Drift in {f}: {abs(filtered_df[f].mean() - df[f].mean()):.2f}")
