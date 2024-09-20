# Use the TensorFlow GPU image as a base
FROM tensorflow/tensorflow:2.13.0-gpu

# Install Python venv and other necessary packages
RUN apt-get update && \
    apt-get install -y wget python3-venv libgl1-mesa-glx vim

RUN python3 -m pip install --upgrade pip && \
    pip install \
    absl-py \
    gin-config \
    gym \
    tf-agents \
    tensorflow \
    scipy \
    pybullet \
    matplotlib \
    tqdm \
    opencv-python

# Copy your code into the container
RUN git clone https://github.com/your-username/your-repo.git /app

# Set the working directory
WORKDIR /app

# Add /app to PYTHONPATH in .bashrc
RUN echo 'export PYTHONPATH=$PYTHONPATH:/app' >> ~/.bashrc
