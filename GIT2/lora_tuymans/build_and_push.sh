#!/bin/bash

# Build a push skript pro LoRA Tuymans Cursor verzi
set -e

# Konfigurace
IMAGE_NAME="mulenmara1505/lora_tuymans_cursor"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🐳 Building Docker image: ${FULL_IMAGE_NAME}"

# Build image pro AMD64 (RunPod kompatibilní)
docker build --platform linux/amd64 -t ${FULL_IMAGE_NAME} .

echo "✅ Build dokončen"

# Test image lokálně (volitelné)
echo "🧪 Testuji image lokálně..."
docker run --rm -d --name test_lora_tuymans -p 8501:8501 ${FULL_IMAGE_NAME}
sleep 10

# Kontrola health
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Health check prošel"
    docker stop test_lora_tuymans
else
    echo "❌ Health check selhal"
    docker stop test_lora_tuymans
    exit 1
fi

echo "🚀 Pushuji image do Docker Hub..."

# Push do Docker Hub
docker push ${FULL_IMAGE_NAME}

echo "✅ Image úspěšně pushnut do Docker Hub: ${FULL_IMAGE_NAME}"

# Zobrazení informací o image
echo "📊 Informace o image:"
docker images ${FULL_IMAGE_NAME}

echo "🎉 Hotovo! Image je dostupný na Docker Hub"
echo "🔗 https://hub.docker.com/r/mulenmara1505/lora_tuymans_cursor"
