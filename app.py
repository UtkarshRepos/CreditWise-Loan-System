import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("loan_approval_data.csv")

# -------------------------------
# Handle missing values
# -------------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Feature Engineering
# -------------------------------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
y = df["Loan_Approved"]

# Identify numeric/categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Preprocessing
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_transformed, y_train)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("CreditWise Loan Approval Prediction")
st.write("Enter applicant details below:")

# -------------------------------
# User inputs
# -------------------------------
user_input = {}

# Numeric inputs
for col in numeric_cols:
    if col in ["DTI_Ratio_sq", "Credit_Score_sq", "Applicant_Income_log"]:
        # calculate from main fields
        if col == "DTI_Ratio_sq":
            dti = st.number_input("DTI Ratio", 0.0, 1.0, 0.3, step=0.01)
            user_input[col] = dti ** 2
        elif col == "Credit_Score_sq":
            cs = st.number_input("Credit Score", 300, 850, 700)
            user_input[col] = cs ** 2
        elif col == "Applicant_Income_log":
            income = st.number_input("Applicant Income", 0, 1000000, 50000)
            user_input[col] = np.log1p(income)
    else:
        # use median as default
        user_input[col] = X[col].median()

# Categorical inputs: use mode as default
for col in categorical_cols:
    user_input[col] = X[col].mode()[0]

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Transform and predict
user_transformed = preprocessor.transform(user_df)

if st.button("Predict Loan Approval"):
    prediction = model.predict(user_transformed)
    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")
