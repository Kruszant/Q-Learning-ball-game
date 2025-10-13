# Use a Python 3.10 base image built on Ubuntu 22.04
FROM python:3.10

# 1. Install system dependencies required for Pygame (SDL libraries)
# Ubuntu uses different package names than Debian/Buster, but the goal is the same.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        libsdl2-ttf-dev \
    # Clean up cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the entire project code into the container
COPY . .

# 5. Define the command to run the application
CMD ["python", "qlearning.py"]
