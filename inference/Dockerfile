# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY inference.py .
COPY . /app

# Install any needed packages specified in inference_requirements.txt
RUN pip install --no-cache-dir -r inference_requirements.txt

# Run the application
CMD ["python", "inference.py"]