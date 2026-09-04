#!/bin/bash
set -e

echo "── Clinical Risk Scoring — Docker + Kubernetes Deploy ──"

# Start Minikube
echo "Step 1: Starting Minikube..."
minikube start --driver=docker --memory=4096 --cpus=2

# Use Minikube's Docker daemon so images are available in-cluster
echo "Step 2: Pointing Docker to Minikube..."
eval $(minikube docker-env)

# Build the Docker image
echo "Step 3: Building Docker image..."
docker build -t clinical-risk-api:latest -f docker/Dockerfile .

# Verify image built
docker images | grep clinical-risk-api

# Apply Kubernetes manifests
echo "Step 4: Applying K8s manifests..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for rollout
echo "Step 5: Waiting for deployment..."
kubectl rollout status deployment/clinical-risk-api --timeout=120s

# Show status
echo "── Deployment complete ──"
kubectl get pods
kubectl get services
echo ""
echo "API URL:"
minikube service clinical-risk-api-service --url
