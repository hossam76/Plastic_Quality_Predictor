
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set expected feature order (must match training)
expected_features = [
    "ZUx - Cycle time",
    "ZDx - Plasticizing time",
    "CPn - Screw position at the end of hold pressure",
    "SVo - Shot volume",
    "Torque_Efficiency",
    "Filling_Speed"
]

# Label map for class interpretation
label_map = {
    1: "Waste",
    2: "Acceptable",
    3: "Target",
    4: "  Inefficient"
}

# Streamlit setup
st.set_page_config(page_title="Plastic Quality Predictor", layout="wide")
st.title("🔍 Plastic Injection Moulding – Quality Predictor Dashboard")
st.markdown("Welcome! Use the tabs below to run predictions and review your model.")

st.sidebar.title("Plastic Injection Moulding- Quality Class Predictor")
st.sidebar.info("Navigate sections from the tabs above to use the model.")
st.sidebar.info("""
**Features:**
- Random Forest Classifier with 91.54% cross-validation accuracy
- Selected Parameters: Cycle time, Plasticizing time, Shot volume, Screw position, Torque Efficiency, Filling Speed
- Streamlit Interface for real-time single and batch predictions
- Model Performance Insights with visual analytics
- Exportable Results for decision-making and traceability
""")

tab1, tab2, tab3 = st.tabs(["📈 Single Prediction", "📁 Batch Prediction", "📘 Model Info"])

# ----- Single Prediction Tab -----
with tab1:

    col1, col2 = st.columns(2)
    features = {}
    st.subheader("📈 Single Prediction")
    with col1:
        features["ZUx - Cycle time"] = st.number_input("Cycle Time (ZUx)", min_value=0.0, value=2.0)
        features["CPn - Screw position at end of hold"] = st.number_input("Screw Position (CPn)", min_value=0.0, value=0.5)
        features["Torque_Efficiency"] = st.number_input("Torque Efficiency", min_value=0.0, value=1.2)
    with col2:
        features["ZDx - Plasticizing time"] = st.number_input("Plasticizing Time (ZDx)", min_value=0.0, value=1.0)
        features["SVo - Shot volume"] = st.number_input("Shot Volume (SVo)", min_value=0.0, value=20.0)
        features["Filling_Speed"] = st.number_input("Filling Speed", min_value=0.0, value=3.5)

    if st.button("Predict Quality Class"):
        input_data = np.array([list(features.values())]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        pred_label = label_map.get(int(prediction), "Unknown")
        st.success(f"🧠 **Predicted Quality Class:** {int(prediction)} — {pred_label}")

# ----- Batch Prediction Tab -----
with tab2:
    st.subheader("📁 Upload CSV for Batch Prediction")
    st.info("Upload a CSV file with the same 6 input columns used above.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data Preview", data.head())

        try:
            data_for_prediction = data[expected_features]
        except KeyError as e:
            st.error(f"❌ Missing required columns: {e}")
        else:
            scaled = scaler.transform(data_for_prediction)
            predictions = model.predict(scaled)
            data['Predicted Quality'] = predictions
            data['Predicted Label'] = data['Predicted Quality'].map(label_map)

            st.success("✅ Predictions completed.")
            st.write("📄 Predictions:", data.head())

            st.subheader("📊 Class-wise Prediction Count")
            class_counts = data['Predicted Quality'].map(label_map).value_counts().sort_index()
            st.bar_chart(class_counts)

            if 'true_quality' in data.columns:
                st.subheader("🧩 Confusion Matrix (True vs Predicted)")
                cm = confusion_matrix(data['true_quality'], data['Predicted Quality'])
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap="Blues", colorbar=False)
                st.pyplot(fig)

            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Results CSV", data=csv_buffer.getvalue(), file_name="predicted_output.csv", mime="text/csv")

# ----- Model Info Tab -----
with tab3:
    st.subheader("📘 Model Information")
    st.write(model)
    if hasattr(model, 'feature_importances_'):
        with st.expander("🔍 View Feature Importances"):
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
