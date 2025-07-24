# MedSegText

MedSegText is a fully cloud-native machine learning pipeline for automated image segmentation and clinical report findings generation from CT scan images. The system is designed for scalable, secure, and reproducible deployment on AWS. It can also be run entirely locally for research and testing.

## 🎥 [Demo Video](https://drive.google.com/file/d/1IVxGnhWPBihEimFN7vuS-nTNQMF_idOr/view?usp=sharing)

---

## **Project Architecture**

- **Model:** Trained deep learning model (joint segmentation + text) deployed on **AWS SageMaker**
- **API:** FastAPI application served via **AWS Lambda** (serverless) and **API Gateway**
- **Storage:** **Amazon S3** for image storage, **DynamoDB** for metadata and results
- **UI:** Streamlit web app containerized and deployed on **AWS ECS Fargate**
- **Data:** MosMedData+ (2700+ CT scan images with text annotation) 
Refer Sample test images folder for examples; full dataset source provided via Google Drive (see below)

---

### **System Architecture Diagram**

![MedSegText Architecture Diagram](arch%20diagrams/MedSegText.png)

> For detailed model architecture, training methodology, and results, see [medsegtext_final_paper.pdf](https://github.com/Tamilarasee/MMI_Unet_Lesion_Segmentation/blob/MedSegText/MedSegText_Final_Paper.pdf).

---

## **How to Use**

All major components are organized in separate folders, each with its own README for setup and deployment:

- **SageMaker Model Deployment:**  
  See [`sagemaker_model_test_container/README_sagemaker_model.md`](sagemaker_model_test_container/README_sagemaker_model.md)  
  *Includes local testing, model weights download, and SageMaker deployment steps.*
> **Note:**  
> This setup alone is enough to try out the core model functionality locally, without needing to deploy the API or UI.

- **API (FastAPI + Lambda):**  
  See [`api/README_api_deployment.md`](api/README_api_deployment.md)  
  *Instructions for packaging, deploying, and configuring the backend API.*

- **UI (Streamlit):**  
  See [`ui/README_ui_deployment.md`](ui/README_ui_deployment.md)  
  *How to run locally or deploy to ECS Fargate.*

- **Model Weights:**  
  Download from [Google Drive](https://drive.google.com/drive/folders/1IaPhul78fYu17qwMLLQrHJeY5p-MqWuo?usp=drive_link)

- **Training:**  
  Use the files in the `MedsegText Training files/` folder and run `main_med_seg_text_exp2.ipynb` for full training pipeline.

- **Complete Data:**   -download from [Google Drive](https://drive.google.com/file/d/16wlFbUgWVwuTq4LFdoWw2uhIqxW6UW9M/view)

---

## Questions or Feedback?

You’re welcome to [open an issue](../../issues) or [connect with me on LinkedIn](https://www.linkedin.com/in/tamilarasee/) for any questions or feedback.

