#!/usr/bin/env bash
# =============================================================================
#  Qwen3.8-27B on K100-AI — container creation script
#  官方镜像 + 官方推荐的容器启动参数。
# =============================================================================
set -euo pipefail

IMAGE="harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828"
NAME="qwen38"

docker pull "$IMAGE"

docker run -it \
    --shm-size 60g \
    --network=host \
    --name "$NAME" \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /data:/data1 \
    "$IMAGE"

# 进入容器并启动服务：
#   docker start qwen38
#   docker exec -it qwen38 bash
#   cd /models/lipf
#   bash config/serve.sh
