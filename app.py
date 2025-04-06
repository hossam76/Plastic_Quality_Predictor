
import streamlit as st
st.set_page_config(page_title="Plastic Quality Predictor", layout="wide")

#importing libraries
import pandas as pd
import joblib
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



# Load trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Expected features used in model training
expected_features = [
    "ZUx - Cycle time",
    "ZDx - Plasticizing time",
    "APVs - Specific injection pressure peak value",
    "time_to_fill",
    "Filling_Speed",
    "SVo - Shot volume",
    "Mm - Torque mean value current cycle"
]

# Map predicted labels to class names
label_map = {
    1: "Waste",
    2: "Acceptable",
    3: "Target",
    4: "Inefficient"
}

# Custom CSS to style the st.sidebar.info() messages
custom_css = """
<style>
    /* Target the st.sidebar.info() alert boxes */
    [data-testid="stSidebar"] .stAlert {
        background-color: #e0f7fa; /* Light cyan background */
        color: #006064; /* Dark cyan text */
        border-left: 5px solid #004d40; /* Darker cyan left border */
    }
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)



# Streamlit UI setup
st.title("🔍 Plastic Injection Moulding – Quality Predictor Dashboard")
st.markdown("Welcome! Use the tabs below to run predictions and review your model.")


st.sidebar.title("Plastic Injection Moulding – Quality Class Predictor")

st.sidebar.markdown("""
    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 10px; color: #0c5460;">
        Navigate sections from the tabs above to use the model.
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 10px; color: #0c5460;">
        <strong>Features:</strong>
        <ul>
            <li>Random Forest Classifier with 93.17% cross-validation accuracy</li>
            <li>Selected Parameters: Cycle time, Plasticizing time, Shot volume, Screw position, Torque Efficiency, Filling Speed</li>
            <li>Streamlit Interface for real-time single and batch predictions</li>
            <li>Model Performance Insights with visual analytics</li>
            <li>Exportable Results for decision-making and traceability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["📈 Single Prediction", "📁 Batch Prediction", "📘 Model Info"])

# ----- Single Prediction Tab -----
with tab1:
    st.subheader("📈 Single Quality Prediction")
    col1, col2 = st.columns(2)
    features = {}

    with col1:
        features["ZUx - Cycle time"] = st.number_input("Cycle Time (ZUx)", min_value=0.0, value=2.0)
        features["APVs - Specific injection pressure peak value"] = st.number_input("Injection Pressure Peak (APVs)", min_value=0.0, value=150.0)
        features["Filling_Speed"] = st.number_input("Filling Speed", min_value=0.0, value=3.5)
        features["Mm - Torque mean value current cycle"] = st.number_input("Torque Mean (Mm)", min_value=0.0, value=0.8)

    with col2:
        features["ZDx - Plasticizing time"] = st.number_input("Plasticizing Time (ZDx)", min_value=0.0, value=1.0)
        features["time_to_fill"] = st.number_input("Time to Fill", min_value=0.0, value=1.5)
        features["SVo - Shot volume"] = st.number_input("Shot Volume (SVo)", min_value=0.0, value=20.0)

    if st.button("🔍 Predict Quality Class"):
        input_data = np.array([list(features.values())])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        pred_label = label_map.get(int(prediction), "Unknown")
        st.success(f"🧠 **Predicted Quality Class:** {int(prediction)} — {pred_label}")

# ----- Batch Prediction Tab -----
with tab2:
    st.subheader("📁 Upload CSV for Batch Prediction")
    st.info("Ensure your CSV file includes the following columns:")
    st.code("\n".join(expected_features))

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📋 Preview of Uploaded Data", data.head())

        try:
            data_for_prediction = data[expected_features]
            scaled_data = scaler.transform(data_for_prediction)
            predictions = model.predict(scaled_data)
            data['Predicted Quality'] = predictions
            data['Predicted Label'] = data['Predicted Quality'].map(label_map)

            st.success("✅ Predictions completed successfully.")
            st.write("🔎 Prediction Results:", data.head())

            st.subheader("📊 Class Distribution")
            class_counts = data['Predicted Label'].value_counts().sort_index()
            st.bar_chart(class_counts)

            if 'true_quality' in data.columns:
                st.subheader("🧩 Confusion Matrix (True vs Predicted)")
                cm = confusion_matrix(data['true_quality'], data['Predicted Quality'])
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap="Blues", colorbar=False)
                st.pyplot(fig)

            # Downloadable predictions
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Results CSV", data=csv_buffer.getvalue(), file_name="predicted_output.csv", mime="text/csv")

        except KeyError as e:
            st.error(f"❌ Missing required columns: {e}")

# ----- Model Info Tab -----
with tab3:
    st.subheader("📘 Model Information")
    st.write(model)

    if hasattr(model, 'feature_importances_'):
        with st.expander("📊 Feature Importances"):
            importance_df = pd.DataFrame({
                'Feature': expected_features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)

            fig, ax = plt.subplots()
            importance_df.plot(kind='barh', x='Feature', y='Importance', ax=ax, legend=False, color='#4CAF50')
            ax.set_title("Feature Importances", fontsize=14)
            ax.invert_yaxis()
            st.pyplot(fig)

    st.subheader("📊 Test Set Confusion Matrix")
    test_file = st.file_uploader("Upload test CSV with 'true_quality' column", type=["csv"], key="test_csv")
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        try:
            X_test = test_df[expected_features]
            y_test = test_df["true_quality"]
            X_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_scaled)

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap="Oranges", colorbar=False)
            st.pyplot(fig)
        except KeyError as e:
            st.error(f"❌ Missing columns: {e}")

# Footer
st.markdown("---")
st.markdown("📘 *Created for ARI Coursework — Streamlit Dashboard by 4134124*")

