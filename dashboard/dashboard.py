import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Heart Disease Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease_uci.csv")
    return df

df = load_data()

st.sidebar.header("Filter Options")

# Filter by Sex
sex_filter = st.sidebar.selectbox(
    "Select Sex",
    options=["All"] + sorted(df["sex"].unique().tolist())
)

if sex_filter != "All":
    df = df[df["sex"] == sex_filter]

# Filter by Chest Pain Type (cp)
cp_filter = st.sidebar.selectbox(
    "Select Chest Pain Type",
    options=["All"] + sorted(df["cp"].unique().tolist())
)

if cp_filter != "All":
    df = df[df["cp"] == cp_filter]

st.subheader("Summary Metrics")
st.metric("Total Records", len(df))
st.metric("Mean Age", f"{df['age'].mean():.2f}")

st.subheader("Filtered Data Preview")
st.dataframe(df.head())

# Visualization 1: Distribution of Age
st.subheader("Distribution of Age")
fig1, ax1 = plt.subplots()
sns.histplot(df["age"], kde=True, ax=ax1)
st.pyplot(fig1)

# Visualization 2: Distribution of Chest Pain Type
st.subheader("Distribution of Chest Pain Type")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(x=df['cp'], palette='coolwarm', ax=ax2)
plt.title('Distribution of Chest Pain Type (cp)')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
st.pyplot(fig2)

# Visualization 3: Gender Distribution
st.subheader("Gender Distribution")
fig3, ax3 = plt.subplots()
sns.countplot(x="sex", data=df, palette='viridis', ax=ax3)
st.pyplot(fig3)
