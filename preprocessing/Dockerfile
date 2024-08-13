# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# # COPY . .
# # COPY ../data/emotions.csv /app/dataset/
# # COPY preprocessing/preprocess.py /app/
# # COPY data/emotions.csv /data/
# # COPY preprocessing/preprocess_requirements.txt /app/

# # COPY preprocessing/data /mnt/data
# # COPY . /app
# # COPY preprocessing/preprocess_requirements.txt /app/
# COPY preprocessing/preprocess_requirements.txt /app/

# # Install any needed packages specified in preprocess_requirements.txt
# RUN pip install --no-cache-dir -r preprocessing/preprocess_requirements.txt

# COPY preprocessing/preprocess.py /app/

# # Set environment variables (optional, if you want to set defaults)
# ENV INPUT_FILE=/mnt/data/emotions.csv
# ENV OUTPUT_FILE=/tmp/processed_emotions.csv
# ENV S3_BUCKET=my-emotions-data-bucket

# # Make port 5000 available to the world outside this container
# EXPOSE 5000

# # Run the application
# CMD ["python", "/app/preprocessing/preprocess.py"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY preprocessing/preprocess_requirements.txt /app/

# Install any needed packages specified in preprocess_requirements.txt
RUN pip install --no-cache-dir -r /app/preprocess_requirements.txt

# Copy the preprocessing script into the container
COPY preprocessing/preprocess.py /app/

# Set environment variables (optional, if you want to set defaults)
ENV INPUT_FILE=/mnt/data/emotions.csv
ENV OUTPUT_FILE=/tmp/processed_emotions.csv
ENV S3_BUCKET=my-emotions-data-bucket

# Expose the port (though not needed for your current script, it can be used for Flask if you uncomment that part)
EXPOSE 5000

# Run the application
CMD ["python", "/app/preprocess.py"]
