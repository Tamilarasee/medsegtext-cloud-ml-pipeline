# Streamlit UI Deployment Instructions

This guide explains how to build, test, and deploy the Streamlit UI for MedSegText using Docker and AWS ECS.

---

## **Quick Local Run**

If you just want to try the UI locally, you can do so with these two commands (make sure your `.env` file with `API_URL` is ready and placed outside this folder):

```bash
docker build -t medsegtext-ui .
docker run -p 8501:8501 --env-file ../.env medsegtext-ui
```

- This will start the app at [http://localhost:8501](http://localhost:8501).

---

Or, follow the detailed steps below for full deployment and production setup.

---

## **Pre-requisites**

- The Streamlit UI is already built and communicates with the Lambda API.
- **Do NOT hardcode the Lambda API URL.**  
  Always use an environment variable (`API_URL`) for the FastAPI/Lambda endpoint.
- **Keep your `.env` file outside this folder** so it is not copied into the Docker image.

---


### Local Docker Testing
- requirements.txt and dockerfile already given in this folder

**Build the Docker image:**
```bash
docker build -t medsegtext-ui .
```

**Run the Docker container:**
```bash
docker run -p 8501:8501 --env-file ../.env medsegtext-ui
```

---

### 4. Verify Local App

- Visit [http://localhost:8501](http://localhost:8501) to confirm the UI is working and can reach the API.

---

### 5. Deploy to AWS ECS

- Tag and push Docker image to AWS ECR.
- Create an ECS cluster and service to run the container.
- Set the `API_URL` environment variable in the ECS task definition (do **not** hardcode in code or image).
- Expose the service
---

## **Summary**

| Step                | Command/Action                                      |
|---------------------|-----------------------------------------------------|
| Build Docker image  | `docker build -t medsegtext-ui .`                   |
| Run locally         | `docker run -p 8501:8501 --env-file ../.env medsegtext-ui` |
| Deploy to AWS       | Push to ECR, run on ECS, set `API_URL` in ECS task  |

---

**For more details, see the AWS ECS and ECR documentation.**
