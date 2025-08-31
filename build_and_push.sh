#!/bin/bash

# Build a push skript pro Docker image na Docker Hub
# Použití: ./build_and_push.sh [tag]

set -e

# Nastavení pro tuymans_lora repozitář
DOCKER_USERNAME="mulenmara1505"
IMAGE_NAME="lora_tuymans"
TAG="${1:-latest}"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "🐳 Building Docker image: ${FULL_IMAGE_NAME}"
echo "📁 Build context: $(pwd)"
echo "⏰ Start time: $(date)"

# Build Docker image
echo "\n🔨 Building image..."
# Multi-platform build pro macOS a Linux kompatibilitu
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "${FULL_IMAGE_NAME}" \
    --tag "${DOCKER_USERNAME}/${IMAGE_NAME}:gpu" \
    --progress=plain \
    --push \
    .

echo "\n✅ Multi-platform build and push completed successfully!"
echo "📦 Images pushed to Docker Hub:"
echo "   - ${FULL_IMAGE_NAME}"
echo "   - ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"
echo "🌍 Platforms: linux/amd64, linux/arm64 (macOS compatible)"

echo "\n✅ All images pushed successfully!"
echo "🔗 Docker Hub: https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"

echo "\n✅ Script completed at $(date)"