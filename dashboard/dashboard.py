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
# SIDEBAR FILTERS (INTERACTIVE 1 & 2)
# =========================
st.sidebar.header("🔎 Filters")

sex_filter = st.sidebar.selectbox(
    "Select Sex",
    ["All"] + sorted(df["sex"].unique().tolist())
)

cp_filter = st.sidebar.selectbox(
    "Chest Pain Type",
    ["All"] + sorted(df["cp"].unique().tolist())
)

# INTERACTIVE 3: SLIDER
age_range = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (40, 70)
)

# SEARCH BOX
search_age = st.sidebar.number_input(
    "Search Exact Age",
    min_value=int(df["age"].min()),
    max_value=int(df["age"].max()),
    value=int(df["age"].mean())
)

# APPLY FILTERS
filtered_df = df.copy()

if sex_filter != "All":
    filtered_df = filtered_df[filtered_df["sex"] == sex_filter]

if cp_filter != "All":
    filtered_df = filtered_df[filtered_df["cp"] == cp_filter]

filtered_df = filtered_df[
    (filtered_df["age"] >= age_range[0]) &
    (filtered_df["age"] <= age_range[1])
]

# search filter (optional)
search_df = df[df["age"] == search_age]

# =========================
# METRICS
# =========================
st.subheader("📊 Summary Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(filtered_df))
col2.metric("Mean Age", f"{filtered_df['age'].mean():.2f}")
col3.metric("Heart Disease Cases", int(filtered_df['target'].sum()))

# =========================
# SIMPLE PREDICTIVE MODEL (RULE-BASED SCORE)
# =========================
st.subheader("🧠 Risk Prediction (Simple Model)")

def risk_score(row):
    score = 0

    if row["age"] > 55:
        score += 2
    if row["chol"] > 240:
        score += 2
    if row["trestbps"] > 140:
        score += 2
    if row["thalach"] < 120:
        score += 2
    if row["exang"] == 1:
        score += 2

    return score

if len(search_df) > 0:
    person = search_df.iloc[0]
    score = risk_score(person)

    if score <= 2:
        risk = "🟢 Low Risk"
    elif score <= 5:
        risk = "🟠 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.write(f"**Selected Age Group:** {search_age}")
    st.write(f"**Risk Score:** {score}/10")
    st.subheader(f"Prediction: {risk}")

# =========================
# DATA PREVIEW
# =========================
st.subheader("📁 Filtered Data")
st.dataframe(filtered_df.head())

# =========================
# VISUALIZATIONS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["age"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="sex", data=filtered_df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Chest Pain Type")
    fig, ax = plt.subplots()
    sns.countplot(x="cp", data=filtered_df, ax=ax)
    st.pyplot(fig)

    st.subheader("Age vs Max Heart Rate")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="age",
        y="thalach",
        hue="target",
        data=filtered_df,
        ax=ax
    )
    st.pyplot(fig)

# =========================
# HEATMAP
# =========================
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(filtered_df.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
