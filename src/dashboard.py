import streamlit as st
import requests
import pandas as pd
import base64
from PIL import Image
import io

st.set_page_config(page_title="Dysgraphia Screener", layout="wide")

st.sidebar.header("⚙️ Clinical Settings")
st.sidebar.write("""
**Diagnostic Sensitivity:**  
Adjust the risk threshold for flagging a student.  
* **Lower Threshold (e.g., 0.70):** Casts a wider net. Increases *Recall* (catches more potential cases, accepts more false alarms).
* **Higher Threshold (e.g., 0.90):** Stricter criteria. Increases *Precision* (fewer false alarms, but may miss subtle cases).
""")
selected_threshold = st.sidebar.slider("Risk Flag Threshold", min_value=0.50, max_value=0.99, value=0.85, step=0.01)
st.sidebar.markdown("---")

st.title("Dysgraphia Early-Detection Screener")
st.write("Upload a batch of isolated handwriting characters from a single student to generate an aggregate structural risk profile.")

uploaded_files = st.file_uploader(
    "Select Handwriting Images (JPG/PNG)", 
    type=['jpg', 'jpeg', 'png', 'bmp'], 
    accept_multiple_files=True
)

if st.button("Analyze Student Sample") and uploaded_files:
    with st.spinner("Processing hybrid extraction and generating Grad-CAM heatmaps..."):
        
        files_payload = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        data_payload = {"threshold": selected_threshold}
        
        try:
            response = requests.post("http://localhost:8000/screen", files=files_payload, data=data_payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.markdown("---")
                    st.subheader("Screening Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Images Processed", data["images_processed"])
                    col2.metric("90th Percentile Risk Score", f"{data['aggregate_risk_score']:.4f}")
                    
                    if data["flagged_for_review"]:
                        col3.error(f"🚨 FLAGGED FOR REVIEW (≥ {data['threshold_used']:.2f})")
                    else:
                        col3.success(f"✅ TYPICAL PROFILE (< {data['threshold_used']:.2f})")
                        
                    st.markdown("---")
                    st.write("### Clinical Explainability (Grad-CAM & Diagnostics)")
                    st.write("Visual heatmaps and explicit geometric deviations activating the classification model.")
                    
                    sorted_letters = sorted(data["letter_details"], key=lambda x: x["probability"], reverse=True)
                    
                    # Display top 8 highest risk letters
                    cols = st.columns(4)
                    for idx, letter in enumerate(sorted_letters[:8]):
                        img_bytes = base64.b64decode(letter["heatmap_b64"])
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        with cols[idx % 4]:
                            st.image(img, use_container_width=True)
                            st.write(f"**{letter['filename']}**")
                            st.write(f"**Risk:** {letter['probability']:.2f}")
                            
                            # Render the new diagnostic text strings
                            for diag in letter["diagnostics"]:
                                st.caption(f"• {diag}")
                            st.markdown("---")
                            
            else:
                st.error(f"Backend API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect. Ensure your FastAPI server is running on port 8000.")