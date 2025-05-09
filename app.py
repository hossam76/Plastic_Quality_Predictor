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

custom_css = """
<style>
[data-testid="stSidebar"] .stAlert {
    background-color: #f0f4f8; /* Light neutral background */
    color: #0f172a; /* Dark navy text for readability */
    border-left: 4px solid #3b82f6; /* Blue left border for contrast */
    padding: 10px;
    border-radius: 8px;
    font-size: 0.9rem;
}

@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] .stAlert {
        background-color: #1e293b; /* Dark navy background */
        color: #ffffff; /* White text */
        border-left: 4px solid #3b82f6; /* Same blue border */
    }
}
</style>
"""


# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


# Streamlit UI setup
st.title("🔍 Plastic Injection Moulding – Quality Predictor Dashboard")
st.markdown("Welcome! Use the tabs below to run predictions and review your model.")

st.sidebar.title("Plastic Injection Moulding- Quality Class Predictor")
st.sidebar.info("Navigate sections from the tabs above to use the model.")
st.sidebar.info("""
**Features:**
- Random Forest Classifier with 93.17% cross-validation accuracy
- Selected Parameters:Cycle time, Plasticizing time, Specific injection pressure peak value , time to fill, Filling Speed, Shot volume, Torque mean value current cycle
- Streamlit Interface for real-time single and batch predictions
- Model Performance Insights with visual analytics
- Exportable Results for decision-making and traceability
""")

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
    for feature in expected_features:
        st.code(feature)


    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📋 Preview of Uploaded Data", data.head())


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

            st.subheader("📦 Production Quality Summary")
            total = len(data)
            accepted = data['Predicted Quality'].isin([2, 3]).sum()
            scrap = data['Predicted Quality'].isin([1]).sum()
            scrap_rate = scrap / total * 100
            st.metric("Total Items", total)
            st.metric("Accepted", accepted)
            st.metric("Scrap", scrap)
            st.metric("Scrap Rate (%)", f"{scrap_rate:.2f}")

            st.subheader("📊 Class-wise Prediction Count")
            class_counts = data['Predicted Label'].value_counts().sort_index()
            st.bar_chart(class_counts)

            if 'true_quality' in data.columns:
                st.subheader("🧩 Confusion Matrix (True vs Predicted)")
                cm = confusion_matrix(data['true_quality'], data['Predicted Quality'])
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap="Blues", colorbar=False)
                st.pyplot(fig)

                st.subheader("📊 Class-wise Evaluation Metrics")
                report = classification_report(data['true_quality'], data['Predicted Quality'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            else:
                st.warning("True quality labels are not available for evaluation.")

            #downloadable csv
            st.subheader("📥 Download Predictions as CSV")
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Results CSV", data=csv_buffer.getvalue(), file_name="predicted_output.csv", mime="text/csv")



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
st.markdown("📘 *Streamlit Dashboard created by 4134124*")
