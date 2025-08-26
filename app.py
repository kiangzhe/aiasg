import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ------------------------
# Helper Functions
# ------------------------
def validate_input(value, name, min_val=None, max_val=None):
    """Validate user numeric input."""
    if value is None:
        st.error(f"{name} cannot be empty.")
        return False
    if min_val is not None and value < min_val:
        st.error(f"{name} must be at least {min_val}.")
        return False
    if max_val is not None and value > max_val:
        st.error(f"{name} must be at most {max_val}.")
        return False
    return True

def train_models(X_train, y_train):
    """Train Logistic Regression, Random Forest, and SVM models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"{name.replace(' ', '_').lower()}.joblib")

    return models

def load_model(model_name):
    """Load a saved model by name."""
    file_name = f"{model_name.replace(' ', '_').lower()}.joblib"
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.warning(f"Model {model_name} not found. Please train models first.")
        return None

# ------------------------
# Streamlit App Config
# ------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("🩺 Diabetes Prediction Web App")
st.write("Built with **Streamlit** | Predict diabetes using ML models")

# Sidebar menu
menu = ["Overview", "Train Models", "Predict Diabetes", "Data Visualization"]
choice = st.sidebar.selectbox("Menu", menu)

# Load Dataset
df = pd.read_csv("diabetes.csv")

# ------------------------
# Menu: Overview
# ------------------------
if choice == "Overview":
    st.subheader("Dataset Overview")
    st.write("Preview of Dataset:")
    st.write("### Column Descriptions")
    st.markdown("""
    **Pregnancies**: Number of times pregnant  
    **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test  
    **BloodPressure**: Diastolic blood pressure (mm Hg)  
    **SkinThickness**: Triceps skin fold thickness (mm)  
    **Insulin**: 2-Hour serum insulin (mu U/ml)  
    **BMI**: Body mass index (weight in kg/(height in m)^2)  
    **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic influence)  
    **Age**: Age in years  
    **Outcome**: Class variable (0 = No diabetes, 1 = Diabetes)  
    """)

    # Show first 10 rows of dataset
    st.write("### First 10 Rows of Dataset")
    st.dataframe(df.head(10))

    # Show dataset shape
    st.write(f"**Shape:** {df.shape}")

    #Show null value
    st.write(f"**Null Values:** {df.isnull().sum().sum()}")
    st.write("**Class Distribution:**")
    st.bar_chart(df['Outcome'].value_counts())

# ------------------------
# Menu: Train Models
# ------------------------
elif choice == "Train Models":
    st.subheader("Train Machine Learning Models")

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    test_size = st.slider("Test Size (in %)", 10, 40, 20) / 100

    if st.button("Train Models"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        joblib.dump(scaler, "scaler.joblib")
        models = train_models(X_train, y_train)
        st.success("✅ Models trained and saved successfully!")

        # Display performance metrics for each model
        st.write("### Model Performance on Test Set:")
        for name, model in models.items():
            y_pred = model.predict(X_test)
        
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        
            st.write(f"**{name}**")
            st.write(f"- Accuracy: {acc:.4f}")
            st.write(f"- Precision: {prec:.4f}")
            st.write(f"- Recall: {rec:.4f}")
            st.write(f"- F1-score: {f1:.4f}")
            st.write("---")

# ------------------------
# Menu: Predict Diabetes
# ------------------------
elif choice == "Predict Diabetes":
    st.subheader("Enter Patient Details")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])

    if st.button("Predict"):
        # Validate inputs
        if all([
            validate_input(pregnancies, "Pregnancies", 0),
            validate_input(glucose, "Glucose", 0),
            validate_input(bp, "Blood Pressure", 0),
            validate_input(skin, "Skin Thickness", 0),
            validate_input(insulin, "Insulin", 0),
            validate_input(bmi, "BMI", 0.0),
            validate_input(dpf, "Diabetes Pedigree Function", 0.0),
            validate_input(age, "Age", 1)
        ]):
            model = load_model(model_choice)
            if model and os.path.exists("scaler.joblib"):
                scaler = joblib.load("scaler.joblib")
                input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                input_data = scaler.transform(input_data)
                prediction = model.predict(input_data)[0]
                result = "Diabetic" if prediction == 1 else "Not Diabetic"
                st.success(f"Prediction: **{result}**")
            else:
                st.error("Model or scaler not found. Please train the models first.")

# ------------------------
# Menu: Data Visualization
# ------------------------
elif choice == "Data Visualization":
    st.subheader("Correlation Matrix of Features")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Model Performance Comparison")

    if os.path.exists("scaler.joblib"):
        scaler = joblib.load("scaler.joblib")
    else:
        st.error("Scaler not found. Please train models first.")
        st.stop()

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.transform(X)

    model_names = ["Logistic Regression", "Random Forest", "SVM"]
    results = []

    for name in model_names:
        model = load_model(name)
        if model:
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1})

    if results:
        results_df = pd.DataFrame(results)
        st.write("### Model Metrics")
        st.dataframe(results_df)

        st.write("### Comparison of Classification Methods")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        melted_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        sns.barplot(data=melted_df, x='Metric', y='Score', hue='Model', ax=ax2)
        ax2.set_title('Model Performance Comparison')
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)
    else:
        st.warning("No models found. Please train models first.")
