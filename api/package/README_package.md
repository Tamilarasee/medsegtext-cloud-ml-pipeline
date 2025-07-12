This package folder must contain all packages needed to run the FASTAPI app, built to be compatible with AWS Lambda (Python 3.9).

**How to prepare this package folder:**

1. Create a Docker environment locally using the AWS ECR image for Python 3.9 to ensure Linux-compatible dependencies:
    ```sh
    docker run --rm -v ${PWD}:/var/task public.ecr.aws/sam/build-python3.9 \
    ```
2. Install the requirements.txt (found in /api) in this folder /package - preparation of the requirements doc was done by troubleshooting iteratively to sync for Python 3.9
    ```sh
    pip install -r requirements.txt -t package
    ```
3. zip this package folder for installation of package zip file in Lambda using below or manually 7-zip
    ```sh
    cd package
    zip -r ../package.zip .
    ```
4. Upload package.zip to AWS Lambda
5. For local testing with docker, update .env file with below
    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    AWS_DEFAULT_REGION=us-east-1

    SAGEMAKER_ENDPOINT=medsegtext-inference
    S3_BUCKET=medsegtext-input-output-images
    DYNAMO_TABLE=Scan-Results
