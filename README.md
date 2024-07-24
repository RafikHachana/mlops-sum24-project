# MLOps Project - Video Game Playtime Prediction
Project Repository for the MLOps course project, Summer Semester 2024, Innopolis University.

## CI Pipeline Status

| Code testing                                                 | Model testing                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Code testing](https://github.com/RafikHachana/mlops-sum24-project/actions/workflows/test-code.yml/badge.svg)](https://github.com/RafikHachana/mlops-sum24-project/actions/workflows/test-code.yml) | [![Model Validation](https://github.com/RafikHachana/mlops-sum24-project/actions/workflows/validate-model.yml/badge.svg)](https://github.com/RafikHachana/mlops-sum24-project/actions/workflows/validate-model.yml) |

## Deployment guide

### Using our custom Flask API

1. Install requirements

   ```
   pip install -r requirements.txt
   ```

2. Create a deployment of the model in the `api` directory (This will also create a Dockerfile for direct MLflow deployment):

   ```
   MLFLOW_TRACKING_URI="http://localhost:5000" mlflow models generate-dockerfile --model-uri models:/model_1@champion --env-manager local -d api
   ```

3. Run the API:

   ```bash
   python3 api/app.py
   ```

4. Access the API locally at `http://localhost:5001/`

### Deploy a docker container

Here we deploy the model as a docker image to Dockerhub, so it can be pulled and executed as a container on other machines or cloud services:

```bash
set -a
# Put your Dockerhub credentials in the environment
export DOCKERHUB_USERNAME=<your-username>
export DOCKERHUB_PASSWORD=<your-password>

bash scripts/deploy_docker.sh
```

### Run the Gradio UI

1. Make sure that the Flask API is running (refer to the instructions above)

2. On the same machine, run:

   ```bash
   python3 src/app.py
   ```

> Note: If you encounter dependency issues (library version mismatches or import error with the Pydantic library), please execute the Gradio UI in another Python virtual environment, as its dependencies may not be compatible with the other dependencies of this project.
