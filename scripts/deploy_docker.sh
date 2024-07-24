#!/bin/bash

# Define variables
IMAGE_NAME="rafikhachana/mlops_deployment"
CONTAINER_NAME="mlops_model"
HEALTHCHECK_URL="http://localhost:8080/ping"
set -a
echo "Updating Dockerfile and directory with latest model..."
export MLFLOW_TRACKING_URI="http://localhost:5000"
mlflow models generate-dockerfile --model-uri models:/model_1@champion --env-manager local -d api
# Build the Docker image
echo "Building Docker image..."
cd api
docker build -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container..."
docker rm -f $CONTAINER_NAME
docker run --rm -d --name $CONTAINER_NAME -p 8080:8080 $IMAGE_NAME

# Wait for the service to be up
echo "Waiting for the service to be up..."
sleep 20

# Check the health of the service
echo "Checking the health of the service..."
curl -s -w "%{http_code}" $HEALTHCHECK_URL
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $HEALTHCHECK_URL)

if [ $HEALTH_STATUS -eq 200 ]; then
    echo "Service is healthy!"
else
    echo "Service is not healthy. Exiting..."
    docker logs $CONTAINER_NAME
    exit 1
fi

# Push the Docker image to Dockerhub
echo "Pushing Docker image to Dockerhub..."
echo $DOCKERHUB_PASSWORD | docker login -u $DOCKERHUB_USERNAME --password-stdin
docker push $IMAGE_NAME

echo "Deployment completed successfully!"