# download_hf_model.py
from transformers import AutoModel, AutoTokenizer
import os

def download_assets():
    """
    Downloads the ConvNeXt model and the CXR-BERT tokenizer from Hugging Face 
    and saves them locally into the sagemaker_model_test_container directory.
    """
    # --- ConvNeXt Model ---
    model_name = "facebook/convnext-tiny-224"
    model_output_dir = os.path.join("sagemaker_model_test_container", "convnext_model")
    
    if os.path.exists(model_output_dir):
        print(f"✅ Model directory '{model_output_dir}' already exists. Skipping.")
    else:
        print(f"Downloading model '{model_name}' to '{model_output_dir}'...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(model_output_dir)
        print("✅ Model downloaded successfully!")

    # --- CXR-BERT Tokenizer ---
    tokenizer_name = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer_output_dir = os.path.join("sagemaker_model_test_container", "tokenizer")

    if os.path.exists(tokenizer_output_dir):
        print(f"✅ Tokenizer directory '{tokenizer_output_dir}' already exists. Skipping.")
    else:
        print(f"Downloading tokenizer '{tokenizer_name}' to '{tokenizer_output_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.save_pretrained(tokenizer_output_dir)
        print("✅ Tokenizer downloaded successfully!")

if __name__ == "__main__":
    download_assets()