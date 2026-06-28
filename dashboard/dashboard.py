import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide") # Use wide layout for better visualization display

st.title("Heart Disease Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease_uci.csv")
    # Convert categorical columns to appropriate types for better plotting
    df['sex'] = df['sex'].astype('category')
    df['cp'] = df['cp'].astype('category')
    df['fbs'] = df['fbs'].astype('category')
    df['restecg'] = df['restecg'].astype('category')
    df['exang'] = df['exang'].astype('category')
    df['slope'] = df['slope'].astype('category')
    df['ca'] = df['ca'].astype('category')
    df['thal'] = df['thal'].astype('category')
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0).astype('category') # Assuming 'num' is the target variable
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
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Mean Age", f"{df['age'].mean():.2f}")
col3.metric("Patients with Heart Disease", df['target'].sum())

st.subheader("Filtered Data Preview")
st.dataframe(df.head())

# Create two columns for visualizations
col_vis1, col_vis2 = st.columns(2)

with col_vis1:
    # Visualization 1: Distribution of Age
    st.subheader("Distribution of Age")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["age"], kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Age Distribution')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    # Visualization 3: Gender Distribution
    st.subheader("Gender Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.countplot(x="sex", data=df, palette='viridis', ax=ax3)
    ax3.set_title('Gender Distribution')
    ax3.set_xlabel('Sex (0 = Female, 1 = Male)')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)

    # New Visualization 4: Distribution of Cholesterol (chol)
    st.subheader("Distribution of Cholesterol")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["chol"], kde=True, ax=ax4, color='lightcoral')
    ax4.set_title('Cholesterol Distribution')
    ax4.set_xlabel('Cholesterol (mg/dl)')
    ax4.set_ylabel('Count')
    st.pyplot(fig4)

with col_vis2:
    # Visualization 2: Distribution of Chest Pain Type
    st.subheader("Distribution of Chest Pain Type")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(x=df['cp'], palette='coolwarm', ax=ax2)
    ax2.set_title('Distribution of Chest Pain Type')
    ax2.set_xlabel('Chest Pain Type')
    ax2.set_ylabel('Count')
    st.pyplot(fig2)

    # New Visualization 5: Age vs. Max Heart Rate (thalach)
    st.subheader("Age vs. Max Heart Rate")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='age', y='thalach', hue='target', data=df, ax=ax5, palette='coolwarm')
    ax5.set_title('Age vs. Maximum Heart Rate Achieved')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Maximum Heart Rate (thalach)')
    st.pyplot(fig5)

    # New Visualization 6: Distribution of Resting Blood Pressure (trestbps)
    st.subheader("Distribution of Resting Blood Pressure")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["trestbps"], kde=True, ax=ax6, color='mediumseagreen')
    ax6.set_title('Resting Blood Pressure Distribution')
    ax6.set_xlabel('Resting Blood Pressure (trestbps)')
    ax6.set_ylabel('Count')
    st.pyplot(fig6)

# New Visualization 7: Correlation Heatmap
st.subheader("Correlation Heatmap of Numerical Features")
fig7, ax7 = plt.subplots(figsize=(10, 8))
# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=['number'])
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax7)
ax7.set_title('Correlation Matrix of Numerical Features')
st.pyplot(fig7)
