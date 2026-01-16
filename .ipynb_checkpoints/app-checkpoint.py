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
df = pd.read_csv("loan_approval_data.csv")  # make sure CSV is in same folder

# -------------------------------
# Feature Engineering
# -------------------------------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

# Drop original columns you transformed
X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
y = df["Loan_Approved"]

# -------------------------------
# Identify categorical and numeric columns
# -------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Preprocessing: scale numeric & encode categorical
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop='first'), categorical_cols)
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

# User inputs
user_input = {}
for col in numeric_cols:
    user_input[col] = st.number_input(col, value=float(X[col].median()))

for col in categorical_cols:
    options = X[col].unique().tolist()
    user_input[col] = st.selectbox(col, options)

# Feature engineering for user input
user_input["DTI_Ratio_sq"] = user_input.get("DTI_Ratio", 0) ** 2
user_input["Credit_Score_sq"] = user_input.get("Credit_Score", 0) ** 2
user_input["Applicant_Income_log"] = np.log1p(user_input.get("Applicant_Income", 0))

# Drop original columns that were transformed
for col in ["Credit_Score", "DTI_Ratio", "Applicant_Income"]:
    if col in user_input:
        user_input.pop(col)

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Transform user input
user_transformed = preprocessor.transform(user_df)

# Predict
if st.button("Predict Loan Approval"):
    prediction = model.predict(user_transformed)
    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")
