import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_baseline_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "best_business_threshold.pkl"

SHAP_EXPLAINER_PATH = BASE_DIR / "models" / "shap_explainer.pkl"
SHAP_FEATURE_NAMES_PATH = BASE_DIR / "models" / "shap_feature_names.pkl"
GLOBAL_SHAP_PATH = BASE_DIR / "reports" /"results" /"global_shap_importance.csv"


# -----------------------------
# Load artifacts
# -----------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    best_threshold = joblib.load(THRESHOLD_PATH)

    shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)

    if not hasattr(shap_explainer, "approximate"):
        shap_explainer.approximate = False

    shap_feature_names = joblib.load(SHAP_FEATURE_NAMES_PATH)
    global_shap_importance = pd.read_csv(GLOBAL_SHAP_PATH)

    return (
        model,
        preprocessor,
        best_threshold,
        shap_explainer,
        shap_feature_names,
        global_shap_importance,
    )
(
     model,
     preprocessor,
     best_threshold,
     shap_explainer,
     shap_feature_names,
     global_shap_importance,
 ) = load_artifacts()

# -----------------------------
# Decision logic
# -----------------------------

def assign_risk_band(probability):
    if probability <= 0.05:
        return "Low Risk"
    elif probability < best_threshold:
        return "Moderate Risk"
    elif probability < 0.30:
        return "High Risk"
    else:
        return "Critical Risk"


def assign_decision(risk_band):
    if risk_band == "Low Risk":
        return "Approve"
    elif risk_band in ["Moderate Risk", "High Risk"]:
        return "Manual Review"
    else:
        return "Reject"


def risk_badge(risk_band):
    if risk_band == "Low Risk":
        return "🟢 Low Risk"
    elif risk_band == "Moderate Risk":
        return "🟡 Moderate Risk"
    elif risk_band == "High Risk":
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"


def generate_recommendation(risk_band, decision):
    if decision == "Approve":
        return "Applicant falls within an acceptable risk range and can be considered for approval."

    if decision == "Manual Review":
        if risk_band == "Moderate Risk":
            return "Applicant shows moderate risk. Request additional income or employment verification."
        return "Applicant shows elevated risk. Perform enhanced credit review before approval."

    return "Applicant exceeds the acceptable business risk threshold and should be rejected or escalated."


# -----------------------------
# Feature configuration
# -----------------------------

feature_config = {
    "Age": {"type": "numeric", "min": 18, "max": 80},
    "Income": {"type": "numeric", "min": 0, "max": 5000000},
    "LoanAmount": {"type": "numeric", "min": 0, "max": 5000000},
    "CreditScore": {"type": "numeric", "min": 300, "max": 850},
    "MonthsEmployed": {"type": "numeric", "min": 0, "max": 600},
    "NumCreditLines": {"type": "numeric", "min": 0, "max": 20},
    "InterestRate": {"type": "numeric", "min": 0.0, "max": 40.0},
    "LoanTerm": {"type": "numeric", "min": 12, "max": 120},
    "DTIRatio": {"type": "numeric", "min": 0.0, "max": 1.0},

    "Education": {"type": "categorical", "options": ["High School", "Bachelor's", "Master's", "PhD"]},
    "EmploymentType": {"type": "categorical", "options": ["Full-time", "Part-time", "Self-employed", "Unemployed"]},
    "MaritalStatus": {"type": "categorical", "options": ["Single", "Married", "Divorced"]},
    "HasMortgage": {"type": "categorical", "options": ["Yes", "No"]},
    "HasDependents": {"type": "categorical", "options": ["Yes", "No"]},
    "LoanPurpose": {"type": "categorical", "options": ["Home", "Auto", "Education", "Business", "Other"]},
    "HasCoSigner": {"type": "categorical", "options": ["Yes", "No"]},
}


feature_sections = {
    "👤 Applicant Information": ["Age", "Education", "MaritalStatus"],
    "💰 Financial Information": ["Income", "CreditScore", "DTIRatio", "NumCreditLines"],
    "🏦 Loan Information": ["LoanAmount", "InterestRate", "LoanTerm", "LoanPurpose"],
    "💼 Employment Information": ["EmploymentType", "MonthsEmployed"],
    "📋 Additional Factors": ["HasDependents", "HasMortgage", "HasCoSigner"],
}

DISPLAY_NAMES = {
    "Age": "Age",
    "Income": "Income",
    "LoanAmount": "Loan Amount",
    "CreditScore": "Credit Score",
    "MonthsEmployed": "Months Employed",
    "NumCreditLines": "Number of Credit Lines",
    "InterestRate": "Interest Rate (%)",
    "LoanTerm": "Loan Term (Months)",
    "DTIRatio": "Debt-to-Income Ratio",
    "Education": "Education",
    "EmploymentType": "Employment Type",
    "MaritalStatus": "Marital Status",
    "HasMortgage": "Has Mortgage",
    "HasDependents": "Has Dependents",
    "LoanPurpose": "Loan Purpose",
    "HasCoSigner": "Has Co-Signer"
}

# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="RiskLens",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ RiskLens")

st.markdown(
    "### Calibrated Default Probability Scoring with Business-Driven Decision Thresholds"
)

st.write(
    "Enter applicant details to estimate default probability, assign a risk band, "
    "generate a lending recommendation, and explain the key drivers behind the decision."
)

st.caption(f"Business decision threshold: {best_threshold:.2%}")


# -----------------------------
# Input form
# -----------------------------

with st.form("risk_form"):
    applicant_data = {}

    for section, features in feature_sections.items():
        st.markdown(f"### {section}")
        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            config = feature_config[feature]
            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                if config["type"] == "numeric":
                    applicant_data[feature] = st.number_input(
                        label=DISPLAY_NAMES[feature],
                        min_value=config["min"],
                        max_value=config["max"],
                        value=None,
                        placeholder=f"Enter {DISPLAY_NAMES[feature]}"
                    )
                else:
                    applicant_data[feature] = st.selectbox(
                        label=DISPLAY_NAMES[feature],
                        options=[None] + config["options"],
                        placeholder=f"Select {DISPLAY_NAMES[feature]}"
                    )

        st.markdown("---")

    submitted = st.form_submit_button("Assess Risk")


# -----------------------------
# Prediction + Explanation
# -----------------------------

if submitted:
    missing_fields = [
        feature for feature, value in applicant_data.items()
        if value is None
    ]

    if missing_fields:
        st.error(f"Please fill all required fields: {', '.join(missing_fields)}")

    else:
        applicant = pd.DataFrame([applicant_data])

        applicant_processed = preprocessor.transform(applicant)

        probability = model.predict_proba(applicant_processed)[:, 1][0]

        risk_band = assign_risk_band(probability)
        decision = assign_decision(risk_band)
        recommendation = generate_recommendation(risk_band, decision)

        st.subheader("RiskLens Decision Report")

        c1, c2, c3 = st.columns(3)

        c1.metric("Default Probability", f"{probability:.2%}")
        c2.metric("Risk Band", risk_badge(risk_band))
        c3.metric("Decision", decision)

        st.progress(float(probability))

        st.subheader("Recommendation")
        st.write(recommendation)

        # -----------------------------
        # SHAP explanation
        # -----------------------------

        st.subheader("Why this decision?")

        shap_values = shap_explainer(applicant_processed)
        values = shap_values.values[0]

        explanation_df = pd.DataFrame({
            "Feature": shap_feature_names,
            "Impact": values
        })

        explanation_df["Absolute Impact"] = explanation_df["Impact"].abs()

        # Clean feature names
        def clean_feature_name(feature):
            feature = feature.replace("num__", "")
            feature = feature.replace("cat__", "")
            feature = feature.replace("_", " ")

            name_map = {
                "Age": "Applicant Age",
                "Income": "Income",
                "LoanAmount": "Loan Amount",
                "CreditScore": "Credit Score",
                "MonthsEmployed": "Months Employed",
                "NumCreditLines": "Number of Credit Lines",
                "InterestRate": "Interest Rate",
                "LoanTerm": "Loan Term",
                "DTIRatio": "Debt-to-Income Ratio",
                "HasCoSigner No": "No Co-Signer",
                "HasCoSigner Yes": "Has Co-Signer",
                "HasMortgage No": "No Mortgage",
                "HasMortgage Yes": "Has Mortgage",
                "HasDependents No": "No Dependents",
                "HasDependents Yes": "Has Dependents",
                "EmploymentType Full-time": "Full-time Employment",
                "EmploymentType Part-time": "Part-time Employment",
                "EmploymentType Self-employed": "Self-employed",
                "EmploymentType Unemployed": "Unemployed",
                "MaritalStatus Married": "Married Status",
                "MaritalStatus Single": "Single Status",
                "MaritalStatus Divorced": "Divorced Status",
                "Education High School": "High School Education",
                "Education Bachelor's": "Bachelor's Education",
                "Education Master's": "Master's Education",
                "Education PhD": "PhD Education",
            }

            return name_map.get(feature, feature)


        def group_feature_name(feature):
            feature = feature.replace("num__", "").replace("cat__", "")

            base_map = {
                "Age": "Applicant Age",
                "Income": "Income",
                "LoanAmount": "Loan Amount",
                "CreditScore": "Credit Score",
                "MonthsEmployed": "Months Employed",
                "NumCreditLines": "Number of Credit Lines",
                "InterestRate": "Interest Rate",
                "LoanTerm": "Loan Term",
                "DTIRatio": "Debt-to-Income Ratio",
                "Education": "Education",
                "EmploymentType": "Employment Type",
                "MaritalStatus": "Marital Status",
                "HasMortgage": "Mortgage Status",
                "HasDependents": "Dependent Status",
                "LoanPurpose": "Loan Purpose",
                "HasCoSigner": "Co-Signer Status",
            }

            for raw_name, clean_name in base_map.items():
                if feature.startswith(raw_name):
                    return clean_name

            return feature


        explanation_df["Clean Feature"] = explanation_df["Feature"].apply(group_feature_name)

        grouped_explanation_df = (
            explanation_df
            .groupby("Clean Feature", as_index=False)["Impact"]
            .sum()
        )

        grouped_explanation_df["Absolute Impact"] = grouped_explanation_df["Impact"].abs()

        top_factors = grouped_explanation_df.sort_values(
            "Absolute Impact",
            ascending=False
        ).head(8)

        increasing_risk = top_factors[top_factors["Impact"] > 0]
        reducing_risk = top_factors[top_factors["Impact"] < 0]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🔺 Top Factors Increasing Risk")

            if not increasing_risk.empty:
                for _, row in increasing_risk.iterrows():
                    st.write(
                        f"🔺 **{row['Clean Feature']}** was one of the strongest contributors to elevated default risk "
                        f"`(+{row['Impact']:.4f})`"
                    )
            else:
                st.info("No major factors increased risk for this applicant.")

        with col_b:
            st.markdown("#### 🔻 Top Factors Reducing Risk")

            if not reducing_risk.empty:
                for _, row in reducing_risk.iterrows():
                    st.write(
                        f"🔻 **{row['Clean Feature']}** reduced risk "
                        f"`({row['Impact']:.4f})`"
                    )
            else:
                st.info("No major factors reduced risk for this applicant.")


        st.caption(
            "SHAP values show how much each feature pushed the model prediction upward "
            "toward higher default risk or downward toward lower default risk. "
            "Positive values increase risk; negative values reduce risk."
        )


        # -----------------------------
        # SHAP waterfall plot
        # -----------------------------

        st.subheader("Applicant-Level SHAP Waterfall")

        try:
            shap_explanation = shap.Explanation(
                values=values,
                base_values=shap_values.base_values[0],
                data=applicant_processed[0].toarray()[0]
                if hasattr(applicant_processed[0], "toarray")
                else applicant_processed[0],
                feature_names=[clean_feature_name(f) for f in shap_feature_names]
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, max_display=10, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning("Waterfall plot could not be displayed, but SHAP driver explanation is available above.")
            st.caption(f"Plot error: {e}")

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray;'>

    <b>RiskLens v1.0</b><br>
    Calibrated Default Probability Scoring with Business-Driven Decision Thresholds<br><br>


    Developed by Siddhi Mane

    </div>
    """,
    unsafe_allow_html=True
)