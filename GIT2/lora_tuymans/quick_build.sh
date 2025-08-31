#!/bin/bash

# Rychlý build pro AMD64 (RunPod kompatibilní)
set -e

IMAGE_NAME="mulenmara1505/lora_tuymans_cursor"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🚀 Rychlý build pro AMD64 platformu..."

# Smazání starého image
docker rmi ${FULL_IMAGE_NAME} 2>/dev/null || true

# Build s AMD64 platformou
echo "🐳 Building AMD64 image..."
docker buildx build --platform linux/amd64 -t ${FULL_IMAGE_NAME} --load .

echo "✅ AMD64 build dokončen!"

# Kontrola architektury
echo "🔍 Kontrola architektury..."
docker inspect ${FULL_IMAGE_NAME} | grep -i architecture

echo "🚀 Push do Docker Hub..."
docker push ${FULL_IMAGE_NAME}

echo "🎉 Hotovo! AMD64 image je na Docker Hubu"


