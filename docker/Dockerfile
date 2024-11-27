# Use Ubuntu as the base image
FROM ubuntu:latest

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install necessary dependencies for Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
    curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Set Python 3.9 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip and install ray with the specified dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir "ray[data,train,tune,serve]==2.38.0"

# Set the working directory
WORKDIR /app

# Define the default command
CMD ["python3"]

COPY ../env/rayconf.yaml ./rayconf.yaml
