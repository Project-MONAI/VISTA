# Use an appropriate base image with GPU support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /workspace

# Copy inference script and requirements
COPY infer_cvpr.py /workspace/infer.py
COPY train_cvpr.py /workspace/train.py
COPY update_ckpt.py /workspace/update_ckpt.py
COPY Dockerfile /workspace/Dockerfile
COPY requirements.txt /workspace/
COPY model_epoch40.pth /workspace
# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the prediction script
COPY predict.sh /workspace/predict.sh
RUN chmod +x /workspace/predict.sh

# Set default command
CMD ["/bin/bash"]
