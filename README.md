# MedSegText: End-to-End Cloud-Native Medical Image Segmentation & Report Findings Generation

MedSegText is a fully cloud-native machine learning pipeline for automated lesion segmentation and clinical report findings generation from chest X-ray images. The system is designed for scalable, secure, and reproducible deployment on AWS, but can also be run entirely locally for research and testing.

---

## **Project Overview**
- **Model:** Trained deep learning model (joint segmentation + text) deployed on **AWS SageMaker**
- **API:** FastAPI application served via **AWS Lambda** (serverless) and **API Gateway**
- **Storage:** **Amazon S3** for image storage, **DynamoDB** for metadata and results
- **UI:** Streamlit web app containerized and deployed on **AWS ECS Fargate**
- **Data:** Sample test images included; full dataset source provided via Google Drive (see below)

> **For detailed model architecture, training methodology, and results, see [medsegtext_final_paper.pdf](https://github.com/Tamilarasee/MMI_Unet_Lesion_Segmentation/blob/MedSegText/MedSegText_Final_Paper.pdf).**

---

## **Architecture Summary**
- **End-to-end cloud-native ML pipeline**: All components are designed for scalable, production-grade deployment on AWS.
- **SageMaker**: Hosts the trained MedSegText model for inference.
- **FastAPI + Lambda**: Provides a secure, serverless REST API for model access.
- **S3**: Stores input images and model outputs.
- **DynamoDB**: Stores metadata, predictions, and logs.
- **ECS (Fargate)**: Runs the Streamlit UI for user interaction.

![MedSegText Architecture Diagram](arch diagrams/MedSegText.pdf)

---

## **Folder Structure & Usage**

- **sagemaker_model_test_container/**
  - Minimal files required for SageMaker model inference (code, configs, test image)
  - Use this folder to test or deploy the model endpoint

- **api/**
  - FastAPI implementation for serving the model as a REST API (Lambda-compatible)
  - Contains code and instructions for both local and AWS Lambda deployment

- **ui/**
  - Streamlit UI code for user interaction
  - Includes Dockerfile and instructions for local run or AWS ECS deployment

- **MedSegText Training files/**
  - All scripts, notebooks, and configs required for model training and experimentation
  - Use this folder to retrain or fine-tune the model (see paper for details)

- **Sample test images**
  - Provided for quick testing and demonstration

- **medsegtext_final_paper.pdf**
  - Full technical report: model details, training setup, results, and references

Each of the main folders (**sagemaker_model_test_container**, **api**, **ui**) contains its own README with step-by-step instructions for local testing and AWS deployment.

---

## **How to Use**
- **Run Locally:**
  - You can run the model, API, and UI entirely on your local machine for development or testing.
- **Deploy to AWS:**
  - Each component is ready for production deployment on AWS (see individual READMEs for details).
- **Retrain the Model:**
  - Use the **MedSegText Training files** folder for further training or experimentation.
  - > *"All model training was performed on an NVIDIA GPU for efficient large-scale experimentation. For full details, refer to Section X of the paper."*
- **Data:**
  - Sample test images are included.
  - **Full dataset source:** [Google Drive link](https://drive.google.com/file/d/16wlFbUgWVwuTq4LFdoWw2uhIqxW6UW9M/view?usp=sharing)

---

## **References & Further Information**
- [medsegtext_final_paper.pdf](https://github.com/Tamilarasee/MMI_Unet_Lesion_Segmentation/blob/MedSegText/MedSegText_Final_Paper.pdf) — Full technical report
- [MedSegText Architecture Diagram](arch diagrams/MedSegText.pdf) 
---

For any questions or contributions, please refer to the individual folder READMEs or contact the project maintainer.
