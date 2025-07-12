import torch
from inference import model_fn, input_fn, predict_fn, output_fn
import base64
import json

# 1. Load the model
model = model_fn(".")

# 2. Prepare a test image 
with open("test_image.png", "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")
payload = {"image": image_b64}
request_body = json.dumps(payload)

# 3. Preprocess input
input_tensor = input_fn(request_body, "application/json")

# 4. Run inference
prediction = predict_fn(input_tensor, model)

# 5. Format output
output, content_type = output_fn(prediction, "application/json")

print("Content-Type:", content_type)
print("Output:", output)

import numpy as np
import matplotlib.pyplot as plt

   # Suppose 'output' is your JSON string from output_fn
import json
result = json.loads(output)
mask = np.array(result["segmentation_mask"])

plt.imshow(mask.squeeze(), cmap="gray")
plt.title("Predicted Segmentation Mask")
plt.show()