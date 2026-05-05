import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import os

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Liver Diagnostic System", page_icon="🩺", layout="wide")

# --- 2. CUSTOM CSS FOR CLINICAL THEME ---
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ffffff; text-align: center;}
    .report-text {font-size: 1.1rem; color: #e0e0e0;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE FUNCTIONS ---

@st.cache_resource
def load_ml_model():
    """Loads the ResU-Net model without compiling to avoid custom metric errors."""
    return tf.keras.models.load_model('ResUNET_Liver_Model2.h5', compile=False)

def analyze_tumor_type(mask):
    """
    Morphological Feature Extraction:
    Analyzes shape and size to classify the potential cancer type.
    """
    pixel_count = np.sum(mask > 0.5)
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "Inconclusive", "No distinct mass found for classification."

    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    
    # Advanced logic based on tumor burden and circularity
    if area > 1800:
        return "Hepatocellular Carcinoma (HCC)", "Primary malignant mass detected. Large volume and infiltrative borders suggest HCC."
    elif (perimeter**2) / (4 * np.pi * area + 1e-6) > 1.5:
        return "Metastatic Liver Disease", "Irregular shape and high perimeter-to-area ratio suggest secondary spread (Metastasis)."
    else:
        return "Cavernous Hemangioma (Likely Benign)", "Small, circumscribed lesion detected. Characteristics often align with benign vascular tumors."

def generate_pdf_report(diagnosis, details, pixel_count):
    """Creates a professional clinical PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(40, 40, 120)
    pdf.cell(200, 20, "AI-ASSISTED RADIOLOGY REPORT", ln=True, align='C')
    
    # Stats
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.cell(200, 10, "Technical Summary:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 8, f"- System: ResU-Net Segmentation Pipeline", ln=True)
    pdf.cell(200, 8, f"- Detected Tumor Burden: {pixel_count} Pixels", ln=True)
    
    # Findings
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Clinical Observations:", ln=True)
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 10, f"The AI analysis identified a region of interest consistent with {diagnosis}. {details}")
    
    pdf.ln(30)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(200, 10, "Note: This is an academic project. Not for official clinical use.", ln=True, align='C')
    
    report_file = "Diagnostic_Report.pdf"
    pdf.output(report_file)
    return report_file

# --- 4. MAIN APP LOGIC ---

st.title("🩺 Advanced Liver AI Diagnostic System")
st.write("Upload a 128x128 Axial CT slice for automated tumor segmentation and classification.")

# Initialize Model
try:
    model = load_ml_model()
except Exception as e:
    st.error(f"Model File Not Found! Ensure 'ResUNET_Liver_Model.h5' is in the folder. Error: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload CT Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Processing Layout
    col1, col2 = st.columns(2)
    
    # Preprocess
    raw_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(raw_img)
    img_resized = cv2.resize(img_array, (128, 128))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(np.expand_dims(img_input, axis=0), axis=-1)

    with col1:
        st.subheader("Input CT Scan")
        st.image(img_resized, use_container_width=True, caption="Resized to 128x128")

    with st.spinner("AI analyzing tissue structures..."):
        # Prediction
        prediction_mask = model.predict(img_input)[0]
        # Sharpen mask for display
        binary_mask = (prediction_mask > 0.5).astype(np.float32)
        tumor_pixel_count = np.sum(binary_mask)

    with col2:
        st.subheader("AI Segmentation Mask")
        if tumor_pixel_count > 50:
            st.image(binary_mask, use_container_width=True, caption="Detected Region")
        else:
            st.write("No significant anomalies detected in this slice.")

    st.markdown("---")

    # --- 5. CLINICAL REPORTING ---
    if tumor_pixel_count > 50:
        diagnosis, details = analyze_tumor_type(binary_mask)
        
        st.markdown("### 📋 Clinical Diagnostic Workflow")
        
        # Report Generation
        report_path = generate_pdf_report(diagnosis, details, tumor_pixel_count)
        
        with open(report_path, "rb") as f:
            # The download button is the trigger for the final classification
            download_clicked = st.download_button(
                label="📥 Download Clinical PDF Report",
                data=f,
                file_name="Liver_Diagnostic_Report.pdf",
                mime="application/pdf"
            )

        # Show Final Type Result only after/during report availability
        if download_clicked:
            st.success("Report generated and downloaded.")
            
        st.info("The section below provides the AI classification based on morphological analysis.")
        with st.expander("🔍 View Final AI Pathological Classification", expanded=True):
            st.markdown(f"## **Condition: {diagnosis}**")
            st.write(f"⚠️ **Tumor Detected.** **Pathological Reasoning:** {details}")
            
            # Severity Bar
            severity = min(float(tumor_pixel_count) / 4000, 1.0)
            st.write(f"Tumor Volume Index: {tumor_pixel_count} px")
            st.progress(severity)
    else:
        st.success("✅ **No Tumor Detected.** The model did not find significant anomalies.")

else:
    st.info("Waiting for image upload...")