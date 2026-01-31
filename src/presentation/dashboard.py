
import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Hierarchical IDS Report", layout="wide")

st.title("Hierarchical Network Intrusion Detection System")
st.markdown("### Project Results & Data Manifest")

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Data Manifest", "Model Performance", "Feature Analysis"])

RESULTS_DIR = "results"
MANIFEST_FILE = "data_manifest.md"

if page == "Data Manifest":
    st.header("Training Data Manifest")
    
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest_content = f.read()
        st.markdown(manifest_content)
    else:
        st.error(f"Manifest file not found at {MANIFEST_FILE}")

elif page == "Model Performance":
    st.header("Model Evaluation Results")
    
    # 1. Classification Report
    st.subheader("Stage 1 Classification Report")
    report_path = os.path.join(RESULTS_DIR, "stage1_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.code(report_text, language="text")
    else:
        st.warning("Report file not found.")

    # 2. Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_path = os.path.join(RESULTS_DIR, "stage1_cm.png")
    if os.path.exists(cm_path):
        image = Image.open(cm_path)
        st.image(image, caption="Stage 1 Confusion Matrix", use_container_width=False, width=600)
    else:
        st.warning("Confusion Matrix image not found.")

elif page == "Feature Analysis":
    st.header("Feature Importance Analysis")
    
    # 1. Feature Importance Plot
    fi_img_path = os.path.join(RESULTS_DIR, "stage1_feature_importance.png")
    if os.path.exists(fi_img_path):
        st.image(Image.open(fi_img_path), caption="Top 20 Features", use_container_width=True)
    
    # 2. Feature Data Table
    st.subheader("Detailed Feature Rankings")
    fi_csv_path = os.path.join(RESULTS_DIR, "stage1_feature_importance.csv")
    if os.path.exists(fi_csv_path):
        df = pd.read_csv(fi_csv_path)
        st.dataframe(df.style.highlight_max(axis=0))
    else:
        st.warning("Feature importance CSV not found.")

# Footer removed
