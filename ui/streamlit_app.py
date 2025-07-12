import streamlit as st
import requests
import datetime
import boto3
import re
import os

API_URL = os.environ.get("API_URL")
st.set_page_config(layout="wide", page_title="MedSegText")

def get_presigned_url(s3_url, expiration=3600):
    # Parse s3://bucket/key
    match = re.match(r"s3://([^/]+)/(.+)", s3_url)
    if not match:
        return None
    bucket, key = match.groups()
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiration,
    )

# --- Header ---
st.markdown("""
    <div style='text-align: center;'>        
        <span style='font-size: 2.5rem; font-weight: bold;'>📝 MedSegText</span><br>
        <span style='font-size: 1.3rem;'>AI-powered chest CT scan analysis</span><br>
        <span style='font-size: 1rem; color: #555;'>Upload a scan, Get the Segmentation mask and Radiology findings</span>
    </div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("medsegtext_form"):
        
    # File uploader centered and spanning both columns
    uploaded_file = st.file_uploader("Upload Scan Image", type=["png", "jpg", "jpeg"])
    
    # Two columns for patient and scan info
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex", ["M", "F", "Non-binary"])
    with col2:
        scan_type = st.text_input("Scan Type", value="CT")
        scan_date = st.date_input("Scan Date", value=datetime.date.today())
        scan_time = st.time_input("Scan Time", value=datetime.datetime.now().time())
        scan_datetime = datetime.datetime.combine(scan_date, scan_time).isoformat()
    
    submit = st.form_submit_button("Analyze")

# --- On Submit ---
if submit:
    if not uploaded_file or not patient_id or not scan_datetime:
        st.error("Please fill all required fields and upload an image.")
    else:
        with st.spinner("Analyzing..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {
                "patient_id": patient_id,
                "age": str(age),
                "sex": sex,
                "scan_type": scan_type,
                "scan_datetime": scan_datetime,
            }
            try:
                response = requests.post(
                    f"{API_URL}/medsegtext-predict", files=files, data=data
                )
                if response.status_code == 200:
                    result = response.json()
                    ##st.success("Analysis complete!")
                    # Display images side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                    with col2:
                        mask_url = result.get("output_mask_url")
                        if mask_url and mask_url.startswith("http"):
                            st.image(mask_url, caption="Segmentation Mask", use_container_width=True)
                        elif mask_url and mask_url.startswith("s3://"):
                            presigned_url = get_presigned_url(mask_url)
                            if presigned_url:
                                st.image(presigned_url, caption="Segmentation Mask", use_container_width=True)
                            else:
                                st.info("Segmentation mask saved to S3. Download and view from your AWS console.")
                    # Show findings and metadata below
                    st.markdown(f"**Radiology Finding:** {result.get('radiology_finding', 'N/A')}")
                    st.markdown(f"**Model Version:** {result.get('model_version', 'N/A')}")
                    st.markdown(f"**Inference Duration:** {result.get('inference_duration', 'N/A')}")
                    
                    st.markdown("**DynamoDB:** Patient Record updated")
                    st.markdown(f"**Input Image S3 URL:** {result.get('input_image_url', 'N/A')}")
                    st.markdown(f"**Output Mask S3 URL:** {result.get('output_mask_url', 'N/A')}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
