"""
Heart Disease Prediction Web Application (Streamlit)
---------------------------------------------------
This application provides a clinical interface for real-time heart disease risk assessment,
featuring global model transparency and local patient-level prediction explainability.

Model & Pipeline Background:
- Algorithm: Gradient Boosting Classifier (selected based on 5-fold Stratified CV F1-score: 0.8296 ± 0.0434).
- Preprocessing: StandardScaler encapsulated in a scikit-learn Pipeline (trained strictly on the 70% training split).
- Serialization: Persisted via pickle as 'heart_disease_model.pkl'.
- Explainability: SHAP TreeExplainer computes exact Shapley attributions for individual patient predictions.
- Feature Order (13 total): age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Attempt to import SHAP for local prediction explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==============================================================================
# 1. Feature Specifications & Metadata
# ==============================================================================
FEATURE_NAMES = [
    'age',       # Age in years (integer: 29 - 77+)
    'sex',       # Sex (1 = Male, 0 = Female)
    'cp',        # Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)
    'trestbps',  # Resting Blood Pressure in mm Hg on admission to hospital (integer: 94 - 200+)
    'chol',      # Serum Cholestoral in mg/dl (integer: 126 - 564+)
    'fbs',       # Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)
    'restecg',   # Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)
    'thalach',   # Maximum Heart Rate Achieved during exercise (integer: 71 - 202)
    'exang',     # Exercise Induced Angina (1 = Yes, 0 = No)
    'oldpeak',   # ST depression induced by exercise relative to rest (float: 0.0 - 6.2+)
    'slope',     # Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)
    'ca',        # Number of major vessels colored by flourosopy (0 - 4)
    'thal'       # Thalassemia status (0: Normal/null, 1: Fixed defect, 2: Reversible defect, 3: Other)
]

FEATURE_LABELS = {
    'age': 'Age (years)',
    'sex': 'Sex (0=Female, 1=Male)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting BP (mm Hg)',
    'chol': 'Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120',
    'restecg': 'Resting ECG (0-2)',
    'thalach': 'Max Heart Rate (bpm)',
    'exang': 'Exercise Angina (0=No, 1=Yes)',
    'oldpeak': 'ST Depression (oldpeak)',
    'slope': 'ST Segment Slope (0-2)',
    'ca': 'Major Vessels (ca: 0-4)',
    'thal': 'Thalassemia (thal: 0-3)'
}

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')


# ==============================================================================
# 2. Model & Explainer Loading with Error Handling
# ==============================================================================
@st.cache_resource
def load_model(model_path: str = None):
    """
    Safely loads the trained Pipeline artifact from disk.
    Handles missing, inaccessible, or corrupted model files gracefully.
    """
    if model_path is None:
        default_dir_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
        if os.path.exists(default_dir_path):
            model_path = default_dir_path
        elif os.path.exists('heart_disease_model.pkl'):
            model_path = 'heart_disease_model.pkl'
        else:
            model_path = default_dir_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Please run 'HeartDisease Predictor_v2.ipynb' to train and save the model."
        )

    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to unpickle model from '{model_path}': {e}. "
            "The model file may be corrupted or created with an incompatible Python/scikit-learn version."
        )

    if not hasattr(loaded_model, 'predict'):
        raise TypeError(
            f"The loaded object from '{model_path}' is not a valid scikit-learn estimator/pipeline "
            "(missing 'predict' method)."
        )

    return loaded_model


@st.cache_resource
def get_explainer(_model):
    """
    Initializes and caches a SHAP TreeExplainer on the pipeline's GradientBoostingClassifier.
    """
    if not SHAP_AVAILABLE:
        return None
    try:
        classifier = _model.named_steps['classifier']
        return shap.TreeExplainer(classifier)
    except Exception:
        return None


# ==============================================================================
# 3. Input Validation & Prediction Logic
# ==============================================================================
def validate_inputs(input_dict: dict) -> list[str]:
    """
    Validates clinical inputs for plausibility, non-nullity, and valid numeric types.
    """
    errors = []

    for key in FEATURE_NAMES:
        if key not in input_dict or input_dict[key] is None:
            errors.append(f"Missing value for clinical feature: '{key}'.")
            continue

        val = input_dict[key]
        if not isinstance(val, (int, float, np.number)) or np.isnan(val) or np.isinf(val):
            errors.append(f"Feature '{key}' must be a valid number, got {val}.")

    if input_dict.get('age', 0) < 1 or input_dict.get('age', 0) > 120:
        errors.append("Age must be between 1 and 120 years.")
    if input_dict.get('trestbps', 0) < 40 or input_dict.get('trestbps', 0) > 300:
        errors.append("Resting Blood Pressure must be between 40 and 300 mm Hg.")
    if input_dict.get('chol', 0) < 50 or input_dict.get('chol', 0) > 800:
        errors.append("Serum Cholesterol must be between 50 and 800 mg/dl.")
    if input_dict.get('thalach', 0) < 30 or input_dict.get('thalach', 0) > 260:
        errors.append("Maximum Heart Rate must be between 30 and 260 bpm.")
    if input_dict.get('oldpeak', 0) < 0.0 or input_dict.get('oldpeak', 0) > 15.0:
        errors.append("ST Depression (oldpeak) must be between 0.0 and 15.0.")

    return errors


def predict_heart_disease(model, input_dict: dict):
    """
    Transforms inputs into a DataFrame matching training schema and performs inference.
    """
    features_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)
    prediction = int(model.predict(features_df)[0])

    probability = None
    if hasattr(model, 'predict_proba'):
        prob_array = model.predict_proba(features_df)
        probability = float(prob_array[0][1])

    return prediction, probability


def explain_prediction(model, explainer, input_dict: dict):
    """
    Generates patient-level feature explanations using SHAP values.
    Returns:
        explanation_df: pd.DataFrame detailing patient values and influence
        fig: matplotlib Figure visualizing feature influence (or None if unavailable)
    """
    features_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)
    scaler = model.named_steps['scaler']
    scaled_features = scaler.transform(features_df)

    if explainer is not None and SHAP_AVAILABLE:
        raw_shap = explainer.shap_values(scaled_features)
        shap_values = np.array(raw_shap).squeeze()

        # Build detailed explanation DataFrame
        rows = []
        for feat in FEATURE_NAMES:
            val = input_dict[feat]
            idx = FEATURE_NAMES.index(feat)
            sv = float(shap_values[idx])
            direction = "Increases Risk 🔺" if sv > 0 else "Decreases Risk 🟢"
            rows.append({
                'Feature': FEATURE_LABELS.get(feat, feat),
                'Patient Value': val,
                'SHAP Impact': sv,
                'Absolute Impact': abs(sv),
                'Influence Direction': direction
            })

        explanation_df = pd.DataFrame(rows).sort_values(by='Absolute Impact', ascending=False).reset_index(drop=True)

        # Generate horizontal bar plot for top influential features
        fig, ax = plt.subplots(figsize=(8, 5))
        top_plot_df = explanation_df.iloc[:10].iloc[::-1]  # Top 10 sorted ascending for horizontal bar chart
        bar_colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_plot_df['SHAP Impact']]

        bars = ax.barh(top_plot_df['Feature'], top_plot_df['SHAP Impact'], color=bar_colors, alpha=0.85)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.set_title("Top 10 Feature Contributions to Prediction (SHAP Values)", fontsize=11, fontweight='bold')
        ax.set_xlabel("SHAP Impact on Log-Odds (Red = Higher Heart Disease Risk, Blue = Lower Risk)", fontsize=9)
        plt.tight_layout()

        return explanation_df[['Feature', 'Patient Value', 'Influence Direction', 'SHAP Impact']], fig

    else:
        # Fallback explanation: Display patient input values with basic z-score direction
        means = scaler.mean_
        stds = scaler.scale_
        rows = []
        for i, feat in enumerate(FEATURE_NAMES):
            val = input_dict[feat]
            z_score = (val - means[i]) / (stds[i] + 1e-9)
            rows.append({
                'Feature': FEATURE_LABELS.get(feat, feat),
                'Patient Value': val,
                'Standardized z-score': round(float(z_score), 2)
            })
        return pd.DataFrame(rows), None


# ==============================================================================
# 4. Streamlit User Interface
# ==============================================================================
def main():
    st.title("Heart Disease Prediction Web App")
    st.write("Enter the required clinical metrics below to evaluate risk state.")

    # Load model and explainer
    try:
        model = load_model(MODEL_FILE)
        explainer = get_explainer(model)
    except Exception as err:
        st.error(f"⚠️ **Model Loading Error**: {err}")
        st.info("Ensure that `HeartDisease Predictor_v2.ipynb` has been executed to export `heart_disease_model.pkl`.")
        st.stop()

    # Layout input forms in 3 columns (Preserving original UI structure)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=40, max_value=300, value=130, step=1)
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], index=0)

    with col2:
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0], index=0)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=50, max_value=800, value=240, step=1)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=30, max_value=260, value=150, step=1)

    with col3:
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1], index=0)
        exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1], index=0)

    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=15.0, value=1.0, step=0.1, format="%.2f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2], index=0)
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-4)", [0, 1, 2, 3, 4], index=0)
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect)", [0, 1, 2, 3], index=0)

    if st.button("Predict"):
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': float(oldpeak),
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        # Input validation
        validation_errors = validate_inputs(input_data)
        if validation_errors:
            for err_msg in validation_errors:
                st.warning(f"⚠️ {err_msg}")
            st.stop()

        # Prediction
        try:
            prediction, probability = predict_heart_disease(model, input_data)

            st.write("---")
            if prediction == 0:
                st.success("The model predicts this person is **Healthy**.")
            else:
                st.error("The model predicts signs of **Heart Disease**.")

            if probability is not None:
                st.caption(f"Estimated Risk Probability: **{probability * 100:.1f}%**")

            # Local Prediction Explanation Section
            st.subheader("🔬 Patient Prediction Breakdown & Feature Influence")
            st.write(
                "The breakdown below shows the patient's individual clinical inputs and illustrates how each "
                "variable influenced the Gradient Boosting model's decision."
            )

            explanation_df, fig = explain_prediction(model, explainer, input_data)

            if fig is not None:
                st.pyplot(fig)

            # Display table of feature contributions
            st.dataframe(explanation_df, use_container_width=True)

            # Essential Clinical Disclaimer distinguishing feature influence from causation
            st.info(
                "ℹ️ **Clinical Note on Feature Influence vs. Causation**: "
                "The feature influence values above reflect statistical attributions within the trained "
                "machine learning model (how each biomarker contributed to shifting the model's output). "
                "They represent **predictive associations**, not direct biological causation. "
                "Interventions and treatments must always be determined by qualified healthcare providers based on comprehensive diagnostic context."
            )

        except Exception as pred_err:
            st.error(f"An unexpected error occurred during prediction: {pred_err}")


if __name__ == "__main__":
    main()