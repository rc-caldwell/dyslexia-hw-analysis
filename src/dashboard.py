import streamlit as st
import requests
import pandas as pd
import base64

# Page Configuration
st.set_page_config(page_title="Dysgraphia Screener", layout="wide")
st.title("Dysgraphia Early-Detection Screener")
st.write("Upload a batch of isolated handwriting characters from a single student to generate an aggregate structural risk profile.")

# File Uploader
uploaded_files = st.file_uploader(
    "Select Handwriting Images (JPG/PNG)", 
    type=['jpg', 'jpeg', 'png', 'bmp'], 
    accept_multiple_files=True
)

if st.button("Analyze Student Sample") and uploaded_files:
    with st.spinner("Processing hybrid extraction pipeline..."):
        
        # Format the files for a multipart/form-data POST request
        files_payload = [
            ("files", (file.name, file.getvalue(), file.type)) 
            for file in uploaded_files
        ]
        
        try:
            # Send to the FastAPI endpoint
            response = requests.post("http://localhost:8000/screen", files=files_payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    st.error(data["error"])
                else:
                    # Render the Results
                    st.markdown("---")
                    st.subheader("Screening Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Images Processed", data["images_processed"])
                    col2.metric("90th Percentile Risk Score", f"{data['aggregate_risk_score']:.4f}")
                    
                    # Highlight the clinical flag
                    if data["flagged_for_review"]:
                        col3.error("🚨 FLAGGED FOR REVIEW")
                    else:
                        col3.success("✅ TYPICAL PROFILE")
                        
                    # Plot the individual letter scores
                    st.write("### Character Analysis & Attention Maps")
                    st.write("Heatmaps indicate the areas where the model flagged an anomaly in the letter shape.")

                    grid_cols = st.columns(4)

                    for idx, item in enumerate(data["results"]):
                        with grid_cols[idx % 4]:
                            image_data = base64.b64decode(item["heatmap_b64"])

                            prob_str = f"Risk: {item['probability']:.1%}"

                            st.image(
                                image_data,
                                caption=f"{item['filename']} | {prob_str}",
                                use_container_width=True
                            )
                    
            else:
                st.error(f"Backend API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect. Ensure your FastAPI server is running on port 8000.")
