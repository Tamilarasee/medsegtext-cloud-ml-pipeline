# export_model.py
import torch
import torch.nn as nn
from med_seg_text_model import JointSegTextUNet
from transformers import AutoTokenizer
import os
import numpy as np
import cv2
from data_preprocessing import SegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class InferenceDataset(SegmentationDataset):
    """
    Modified version of SegmentationDataset for inference
    Only processes images, no masks or text needed
    """
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_path)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        # Normalize
        image = image.to(torch.float32) / 255.0
        
        return image

def export_model(model_path, output_path):
    """
    Export the PyTorch model to ONNX format
    
    Args:
        model_path: Path to the .pth file containing model weights
        output_path: Where to save the ONNX model
    """
    # Initialize model with same parameters as training
    model = JointSegTextUNet(
        seg_out_channels=1,
        vocab_size=30524,  # From your tokenizer
        embed_dim=768,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=3072,
        max_text_seq_len=50,
        dropout=0.1,
        pad_token_id=0
    )
    
    # Load trained weights
    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Create dummy input for ONNX export
    # Note: We only need the image input for ONNX export
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Export the model
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_image,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['segmentation_output', 'text_output'],
        # dynamic axes for batch size in input and output
        dynamic_axes={
            'input': {0: 'batch_size'},
            'segmentation_output': {0: 'batch_size'},
            'text_output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")

def test_onnx_model(onnx_path, test_image_path):
    """
    Test the exported ONNX model with a real image
    
    Args:
        onnx_path: Path to the exported ONNX model
        test_image_path: Path to a test image
    """
    import onnxruntime as ort
    
    # Use the same transforms as in your notebook
    val_test_transforms = A.Compose([
        A.Resize(224, 224),
        ToTensorV2()   
    ])
    
    # Create inference dataset
    inference_dataset = InferenceDataset(test_image_path, transform=val_test_transforms)
    
    # Get preprocessed image
    input_tensor = inference_dataset[0]
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Run inference
    outputs = session.run(None, {'input': input_tensor.numpy()})
    print("ONNX model test successful!")
    print(f"Segmentation output shape: {outputs[0].shape}")
    print(f"Text output shape: {outputs[1].shape}")
    
    return outputs

if __name__ == "__main__":
    # Export the model
    model_path = "medsegtext_joint_trained_exp2.pth"
    output_path = "model.onnx"
    
    # Export model
    export_model(model_path, output_path)
    
    # Test with a sample image (you'll need to provide a test image path)
    # test_image_path = "path_to_test_image.jpg"
    # outputs = test_onnx_model(output_path, test_image_path)