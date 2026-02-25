FROM python:3.11-slim

# System dependencies for ComfyUI + video generation
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app && \
    pip install --no-cache-dir -r requirements.txt

# Install AnimateDiff-Evolved custom node
RUN cd /app/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git && \
    cd ComfyUI-AnimateDiff-Evolved && \
    pip install --no-cache-dir -r requirements.txt || true

# Install VideoHelperSuite (for video output nodes — GIF/MP4 encoding)
RUN cd /app/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt || true

# Create model directories and download AnimateDiff SDXL motion model
RUN mkdir -p /app/models/animatediff_models /app/models/animatediff_motion_lora && \
    wget -q -O /app/models/animatediff_models/mm_sdxl_v10_beta.ckpt \
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt" || \
    echo "Warning: AnimateDiff motion model download failed — will need manual download"

# Install IPAdapter Plus custom node (for character consistency via face conditioning)
RUN cd /app/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git && \
    cd ComfyUI_IPAdapter_plus && \
    pip install --no-cache-dir -r requirements.txt || true

# Download IP-Adapter plus-face SDXL model + CLIP vision encoder
RUN mkdir -p /app/models/ipadapter /app/models/clip_vision && \
    wget -q -O /app/models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" && \
    wget -q -O /app/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" || \
    echo "Warning: IP-Adapter model download failed — character consistency will be unavailable"

# Default port
EXPOSE 8188

# Run ComfyUI — listen on all interfaces, no browser launch
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
