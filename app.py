import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config

set_config(transform_output='pandas')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# Title and description
st.title("🫀 Heart Disease Risk Prediction")
st.markdown("""
This application predicts the risk of heart disease based on patient medical data.
Please enter the patient information below to get a prediction.
""")


# Load or train model
@st.cache_resource
def load_model():
    try:
        # Try to load existing model
        with open('heart_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # Train new model if no saved model exists
        st.info("Training model... This may take a moment.")
        
        # Load training data
        df_train = pd.read_csv('heart_train.csv')
        X = df_train.drop(['HeartDisease', 'PatientId'], axis=1)
        y = df_train['HeartDisease']
        
        # Define features
        num_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        cat_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        # Preprocessing pipelines
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', 
                                     sparse_output=False,
                                     min_frequency=0.01))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', num_pipe, num_features),
            ('cat', cat_pipe, cat_features)
        ])
        
        # Create and train model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ))
        ])
        
        model.fit(X, y)
        
        # Save model
        with open('heart_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model

# Load the model
model = load_model()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=50,
    )
    
    sex = st.selectbox(
        "Sex",
        options=["M", "F"],
        format_func=lambda x: "Male" if x == "M" else "Female"
    )
    
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120,
    )
    
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0,
        max_value=600,
        value=200,
    )
    
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )
    
    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150,
    )

with col2:
    
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=["ATA", "NAP", "ASY", "TA"],
        format_func=lambda x: {
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "ASY": "Asymptomatic",
            "TA": "Typical Angina"
        }[x],
    )
    
    resting_ecg = st.selectbox(
        "Resting ECG Result",
        options=["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST-T Wave Abnormality",
            "LVH": "Left Ventricular Hypertrophy"
        }[x],
    )
    
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        options=["N", "Y"],
        format_func=lambda x: "No" if x == "N" else "Yes",
    )
    
    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=-3.0,
        max_value=7.0,
        value=0.0,
        step=0.1,
    )
    
    st_slope = st.selectbox(
        "ST Slope",
        options=["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Upsloping",
            "Flat": "Flat",
            "Down": "Downsloping"
        }[x],
    )

# Prediction button
st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # Make prediction
    with st.spinner('Analyzing patient data...'):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prediction",
            value="High Risk" if prediction == 1 else "Low Risk",
            delta="Positive" if prediction == 1 else "Negative"
        )
    
    with col2:
        st.metric(
            label="Risk Probability",
            value=f"{prediction_proba[1]*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Confidence",
            value=f"{max(prediction_proba)*100:.1f}%"
        )
    
    # Visual indicator
    if prediction == 1:
        st.error("""
        ⚠️ **High Risk of Heart Disease Detected**
        
        This patient shows indicators consistent with increased heart disease risk. 
        Please consult with a cardiologist for further evaluation and treatment planning.
        """)
    else:
        st.success("""
        ✅ **Low Risk of Heart Disease**
        
        Based on the provided data, this patient shows a lower risk profile for heart disease.
        Continue monitoring and maintain healthy lifestyle practices.
        """)
    
    # Display input summary
    with st.expander("📝 View Input Summary"):
        st.dataframe(input_data.T, use_container_width=True)


