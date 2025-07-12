import os
import sys
import json
import base64

# Add the code directory to the path
sys.path.append('/opt/ml/model/code')

try:
    # Import your inference functions
    print("Importing inference module...")
    from inference import model_fn, input_fn, predict_fn, output_fn
    print("Successfully imported inference module")

    # Test model loading
    print("Testing model loading...")
    model = model_fn('/opt/ml/model')
    print("Model loaded successfully!")

    # Test inference with a sample input
    print("Testing inference...")
    with open("test_image.png", "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    sample_input = {
        "image": image_b64
    }

    # Convert to the format expected by your model
    input_data = input_fn(json.dumps(sample_input), 'application/json')

    # Run prediction
    output = predict_fn(input_data, model)

    # Format the output
    result = output_fn(output, 'application/json')

    print("Inference successful!")
    print("Result:", result)
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
