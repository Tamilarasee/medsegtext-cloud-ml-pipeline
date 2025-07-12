# test_endpoint_simple.py
import boto3
import json
import base64
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from botocore.config import Config

# Load environment variables from .env file
load_dotenv()

def visualize_result(result):
    """
    Converts the segmentation mask from the endpoint response into an image 
    and displays it using matplotlib.
    """
    print("\nVisualizing segmentation mask...")

    try:
        # The mask is returned as a list of lists. We convert it to a NumPy array.
        # The model output shape is likely (1, 1, 224, 224). The .squeeze()
        # method removes the extra single dimensions to get a 2D array (224, 224)
        # which is needed for displaying an image.
        seg_mask = np.array(result['segmentation_mask']).squeeze()

        plt.imshow(seg_mask, cmap='gray')
        plt.title("Predicted Segmentation Mask from SageMaker")
        plt.axis('off') # Hide the x and y axes
        plt.show()

    except Exception as e:
        print(f"❌ Could not visualize the mask. Error: {e}")


def test_endpoint():
    # Configure a longer timeout for the client
    # We'll set it to 5 minutes (300 seconds) to be safe
    config = Config(
        read_timeout=300,
        retries={'max_attempts': 0} # Optional: Don't retry on timeout
    )

    # Initialize the SageMaker runtime client with the new config
    runtime = boto3.client(
        'sagemaker-runtime', 
        region_name='us-east-1', 
        config=config
    )
    
    # Read and encode the image
    with open('test_image.png', 'rb') as f:
        image_bytes = f.read()
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare the payload
    payload = {
        "image": image_b64
    }
    
    try:
        # Send request to endpoint
        print("Sending request to SageMaker endpoint...")
        response = runtime.invoke_endpoint(
            EndpointName='medsegtext-inference',
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        print("✅ Success!")
        print("\nGenerated Text:")
        print(result['generated_text'])

        # Instead of just printing the shape, we will now visualize the mask
        visualize_result(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during endpoint invocation: {e}")
        return None

if __name__ == "__main__":
    test_endpoint()