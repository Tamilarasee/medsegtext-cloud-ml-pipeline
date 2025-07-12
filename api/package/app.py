# app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import boto3
import base64
import json
import os
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
from PIL import Image
from io import BytesIO
from botocore.config import Config
from dotenv import load_dotenv
from mangum import Mangum # Used by ASGI apps like FastAPI to interact with AWS Lambda

# Load environment AWS variables from .env file
load_dotenv()

app = FastAPI()

# AWS setup
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "medsegtext-inference")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "medsegtext-input-output-images")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "Scan-Results")

config = Config(
    read_timeout=300,
    retries={'max_attempts': 0}
)
runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=config)
s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMO_TABLE)

def upload_to_s3(file_bytes, key):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes)
    return f"s3://{S3_BUCKET}/{key}"

@app.post("/medsegtext-predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    scan_type: str = Form(...),
    scan_datetime: str = Form(...)  # User-provided scan date/time
):

# when  ui sends requests.post(..., files=files, data=data)---FastAPI’s UploadFile = File(...) and Form(...) parameters are designed to automatically extract the file and form fields
    # 1. Upload input image to S3
    image_bytes = await file.read()
    record_id = str(uuid4())
    upload_timestamp = datetime.now(timezone.utc).isoformat()

    input_key = f"uploads/{patient_id}_{scan_datetime}_{record_id}.png"
    input_image_url = upload_to_s3(image_bytes, input_key)

    # 2. Prepare payload for SageMaker (ONLY the image)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image": image_b64}

    # 3. Call SageMaker endpoint and time it
    inference_start =  datetime.now(timezone.utc)
    response = runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    inference_end = datetime.now(timezone.utc)
    inference_timestamp = inference_end.isoformat()
    inference_duration = (inference_end - inference_start).total_seconds()

    result = json.loads(response["Body"].read().decode())

    # 4. Upload output mask to S3
    seg_mask = result.get("segmentation_mask")
    output_mask_url = None
    if seg_mask is not None:
        mask_array = np.array(seg_mask).squeeze() #remove extra dimensions (e.g., batch/channel).
        mask_img = Image.fromarray((mask_array * 255).astype("uint8"))
        buf = BytesIO() # Create an in-memory buffer
        mask_img.save(buf, format="PNG") # Create an in-memory buffer
        buf.seek(0) # Reset the buffer pointer to the beginning
        output_key = f"outputs/{patient_id}_{scan_datetime}_{record_id}_mask.png"
        output_mask_url = upload_to_s3(buf.read(), output_key)  # sending bytes to s3 with buf.read()

    # 5. Store record in DynamoDB
    item = {
        "record_id": record_id,
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "scan_type": scan_type,
        "scan_datetime": scan_datetime,
        "upload_timestamp": upload_timestamp,
        "inference_timestamp": inference_timestamp,
        "input_image_url": input_image_url,
        "output_mask_url": output_mask_url,
        "radiology_finding": result.get("generated_text"),
        "model_version": "v1.0",
        "inference_duration": f"{inference_duration:.2f}s"
    }
    table.put_item(Item=item) # save to DynamoDB

    # 6. Return result
    return JSONResponse(content=item)

@app.get("/")
def read_root():
    return {"message": "MedsegText API is now running! Use /medsegtext-predict to POST scan image and patient data."}

handler = Mangum(app)