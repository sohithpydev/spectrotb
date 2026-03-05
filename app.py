import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import joblib
import os
import sys

# Add src to path to import helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import baseline_clsa, normalize_tic
from scipy.signal import savgol_filter
from src.features import extract_features, get_feature_names

# --- Configuration & Styling ---
st.set_page_config(page_title="TB/NTM MALDI-TOF Analyzer", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .prediction-tb { color: #d9534f; font-size: 32px; font-weight: bold; padding: 20px; border-radius: 10px; background-color: #f2dede; text-align: center; }
    .prediction-ntm { color: #5cb85c; font-size: 32px; font-weight: bold; padding: 20px; border-radius: 10px; background-color: #dff0d8; text-align: center; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'models', 'hgb_clsa_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def parse_uploaded_file(uploaded_file):
    # Read the text file bytes
    content = uploaded_file.read().decode('utf-8')
    lines = content.split('\n')
    
    mzs, ints = [], []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                mzs.append(float(parts[0]))
                ints.append(float(parts[1]))
            except ValueError:
                continue
                
    if not mzs: return None, None
    return np.array(mzs), np.array(ints)

def plot_step(mz, intensity, title, color, ax, baseline=None):
    ax.plot(mz, intensity, color=color, linewidth=1.5)
    if baseline is not None:
        ax.plot(mz, baseline, color='red', linestyle='--', linewidth=1, label='Estimated Baseline')
        ax.legend(loc='upper right')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)

def analyzer_page(model):
    st.title("🔬 MALDI-TOF Spectrum Analyzer (TB vs NTM)")
    st.write("Upload a raw `.txt` mass spectrum file to visualize real-time CLSA preprocessing and receive an AI-powered diagnostic prediction using the backend HistGradientBoosting model.")
    
    # 2. File Upload
    uploaded_file = st.file_uploader("Upload Spectrum File (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        with st.spinner("Parsing spectrum and executing preprocessing pipeline..."):
            # Load Data
            mz_raw, int_raw = parse_uploaded_file(uploaded_file)
            
            if mz_raw is None or len(mz_raw) == 0:
                st.error("Could not parse numeric m/z and intensity values from the provided file.")
                return
                
            # Filter Mass Range (same as src/data_loader.py)
            mask = (mz_raw >= 3000) & (mz_raw <= 15000)
            mz = mz_raw[mask]
            intensity = int_raw[mask]
            
            # --- Preprocessing Steps ---
            # 1. Smoothing
            int_smoothed = savgol_filter(intensity, window_length=9, polyorder=3)
            
            # 2. CLSA Baseline
            baseline = baseline_clsa(mz, int_smoothed, k=2.0, transform_mz=True)
            int_corrected = int_smoothed - baseline
            int_corrected[int_corrected < 0] = 0
            
            # 3. TIC Normalization
            int_norm = normalize_tic(int_corrected)
            
            # --- Visualizations ---
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<p class='big-font'>Preprocessing Steps Visualization</p>", unsafe_allow_html=True)
            
            fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            
            plot_step(mz, intensity, "1. Raw Spectrum", "gray", axs[0])
            plot_step(mz, int_smoothed, "2. Savitzky-Golay Smoothed Spectrum", "blue", axs[1])
            plot_step(mz, int_smoothed, "3. CLSA Baseline Correction", "purple", axs[2], baseline=baseline)
            axs[2].fill_between(mz, int_corrected, color='purple', alpha=0.3) # Show the resulting isolated peaks
            plot_step(mz, int_norm, "4. TIC Normalized Spectrum", "green", axs[3])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- Feature Extraction & Prediction ---
            with st.spinner("Extracting Biomarkers & Running Model Inference..."):
                feats = extract_features(mz, int_norm)
                X_pred = np.array(feats).reshape(1, -1)
                
                # Predict
                prediction = model.predict(X_pred)[0]
                proba = model.predict_proba(X_pred)[0]
                
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<p class='big-font'>Machine Learning Diagnostics</p>", unsafe_allow_html=True)
            
            # 0 = NTM, 1 = TB
            if prediction == 1:
                st.markdown(f"<div class='prediction-tb'>Diagnosis: Tuberculosis (TB) <br> <span style='font-size: 20px'>Confidence: {proba[1]*100:.2f}%</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-ntm'>Diagnosis: Nontuberculous Mycobacteria (NTM) <br> <span style='font-size: 20px'>Confidence: {proba[0]*100:.2f}%</span></div>", unsafe_allow_html=True)
                
            # Show Extracted Features
            st.write("### Extracted Biomarker Intensities:")
            feature_names = get_feature_names()
            df_feats = pd.DataFrame([feats], columns=feature_names)
            st.dataframe(df_feats.style.format("{:.6f}"))

def contact_page():
    st.title("📬 Contact Us")
    st.markdown("---")
    
    st.markdown("""
    ### Development Team
    This web application and the underlying machine-learning driven clinical pipeline were developed by:
    
    * **Sohith Reddy** (Developer & Ex-Research Intern)  
      Personal Website: https://sohithpydev.github.io/sohith/  
      Email: sohith.bme@gmail.com
    
    * **Professor Wen-Ping Peng** (Principal Investigator)  
      Email: pengw@gms.ndhu.edu.tw
    
    * **Dr. Avinash Patil** (Postdoctoral Researcher)
    
    ### Laboratory
    **Biophysical Mass Spectrometry (BMS) Lab**  
    National Dong Hwa University (NDHU)  
    Hualien, Taiwan  
    
    Lab Members & Research Group:  
    https://faculty.ndhu.edu.tw/~PENGW/members/
    
    ---
    For scientific inquiries, data analysis algorithms, or technical support regarding MALDI-TOF MS spectral preprocessing and machine learning diagnosis, please contact the development team or the BMS laboratory.
    """)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Analyzer Dashboard", "Contact Us"])
    
    if page == "Analyzer Dashboard":
        # 1. Load Model
        model = load_model()
        if model is None:
            st.error("Error: Trained model `hgb_clsa_model.pkl` not found. Please train the backend model first.")
            return
        analyzer_page(model)
    elif page == "Contact Us":
        contact_page()

if __name__ == "__main__":
    main()
