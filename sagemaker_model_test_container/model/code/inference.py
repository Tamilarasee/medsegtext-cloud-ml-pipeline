# inference.py

import torch
from med_seg_text_model import JointSegTextUNet
from seg_decoder import SegmentationDecoder
from text_utils import load_tokenizer
import os

def model_fn(model_dir):
    """
    Load the model AND the tokenizer for SageMaker inference.
    """
    # Construct the path to the downloaded ConvNeXt model files inside the container
    convnext_path = os.path.join(model_dir, "convnext_model")

    # Initialize model with the same parameters as training, plus the local path
    model = JointSegTextUNet(
        seg_out_channels=1,
        embed_dim=768,
        vocab_size=30524, # from tokenizer bio --> 30522+SOS+EOS,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=3072,
        max_text_seq_len=50,
        dropout=0.1,
        pad_token_id=0,
        convnext_model_path=convnext_path # Pass the local path here
    )
    model_path = os.path.join(model_dir, "medsegtext_joint_trained_exp2.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # --- Load Tokenizer ---
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)

    # Return a dictionary containing all model assets
    return {'model': model, 'tokenizer': tokenizer}

import base64
import io
from PIL import Image
import numpy as np
import torch

def input_fn(request_body, request_content_type):
    """
    Preprocess the input for SageMaker inference.
    Expects a JSON with a base64-encoded image.
    """
    import json
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Assume the image is base64-encoded under the key 'image'
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to 224x224
        image = image.resize((224, 224))
        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        # HWC to CHW - PyTorch models expect Channels x Height x Width (CHW).
        image_np = np.transpose(image_np, (2, 0, 1))
        # Convert to torch tensor and add batch dimension
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return image_tensor
    else:
        raise Exception(f"Unsupported content type: {request_content_type}")
    
@torch.no_grad()
def generate_finding(model, tokenizer, image_tensor, device, max_length=50):
    model.eval()
    image_batch = image_tensor.to(device)
    f0 = model.conv_input(image_batch)
    f1 = model.visual_encoder.stage1(f0)
    f2 = model.visual_encoder.stage2(f1)
    f3 = model.visual_encoder.stage3(f2)
    f4 = model.visual_encoder.stage4(f3)
    memory = f4.flatten(2).permute(0, 2, 1)
    memory = model.visual_feature_proj(memory)
    current_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    for _ in range(max_length - 1):
        tgt_embed = model.text_embedding(current_ids)
        tgt_embed_pe = tgt_embed.permute(1, 0, 2)
        tgt_embed_pe = model.positional_encoding(tgt_embed_pe)
        tgt_embed_pe = tgt_embed_pe.permute(1, 0, 2)
        current_seq_len = current_ids.size(1)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(current_seq_len, device=device)
        decoder_output = model.transformer_decoder(
            tgt=tgt_embed_pe,
            memory=memory,
            tgt_mask=tgt_mask
        )
        last_token_logits = model.text_output_layer(decoder_output[:, -1, :])
        predicted_id = torch.argmax(last_token_logits, dim=-1)
        current_ids = torch.cat([current_ids, predicted_id.unsqueeze(0)], dim=1)
        if predicted_id.item() == tokenizer.eos_token_id:
            break
    generated_ids = current_ids[0, 1:].tolist()
    if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
        generated_ids = generated_ids[:-1]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def predict_fn(input_data, model_assets):
    """
    Run inference on the input data using the loaded model.
    """
    # Unpack model assets
    model = model_assets['model']
    tokenizer = model_assets['tokenizer']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)

    # Segmentation
    with torch.no_grad():
        seg_output = model(image=input_data, mode='segmentation')
        seg_mask = torch.sigmoid(seg_output).cpu().numpy()  # Convert to numpy for output

    # Text generation
    text_output = generate_finding(model, tokenizer, input_data, device)

    # Return both outputs
    return {
        "segmentation_mask": seg_mask.tolist(),  # JSON serializable
        "generated_text": text_output
    }

import json

def output_fn(prediction, accept):
    """
    Format the prediction output for the API response.
    """
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    else:
        raise Exception(f"Unsupported Accept type: {accept}")