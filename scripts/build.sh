#!/usr/bin/env bash
# mac-server에서 실행 — API + UI 이미지 빌드 후 registry.internal 푸시
#
# Usage:
#   ssh mac-server "cd ~/Projects/pot-of-greed && git pull && bash scripts/build.sh"
set -euo pipefail

REGISTRY=registry.internal:5000
SHA=$(git rev-parse --short HEAD)

echo "=== pot-of-greed build: $SHA ==="

docker build \
    -t "$REGISTRY/pot-of-greed:$SHA" \
    -t "$REGISTRY/pot-of-greed:latest" \
    . && \
docker push "$REGISTRY/pot-of-greed:$SHA" && \
docker push "$REGISTRY/pot-of-greed:latest"

docker build \
    -t "$REGISTRY/pot-of-greed-ui:$SHA" \
    -t "$REGISTRY/pot-of-greed-ui:latest" \
    -f ui/Dockerfile \
    . && \
docker push "$REGISTRY/pot-of-greed-ui:$SHA" && \
docker push "$REGISTRY/pot-of-greed-ui:latest"

echo "pushed: $REGISTRY/pot-of-greed:$SHA"
echo "pushed: $REGISTRY/pot-of-greed-ui:$SHA"
