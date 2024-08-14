# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY data /app/dataset
COPY preprocess.py .
COPY . /app

# Install any needed packages specified in preprocess_requirements.txt
RUN pip install --no-cache-dir -r preprocess_requirements.txt

# Run the application
CMD ["python", "preprocess.py"]
