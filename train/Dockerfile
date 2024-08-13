# FROM adamrusai/preprocess:latest as preprocess-stage

# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY --from=preprocess-stage /app/dataset /app/processed_train
# COPY . /app
# # COPY /app/data/preprocessed_data.csv /app/data/
# # COPY train/train.py /app/
# # # COPY data/preprocessed_data.csv /data/
# # COPY train/train_requirements.txt /app/

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r train/train_requirements.txt

# # Make port 5000 available to the world outside this container
# EXPOSE 5000

# # Run the application
# CMD ["python", "/app/train/train.py"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the train script and requirements file
COPY train/train_requirements.txt /app/

# Install any needed packages specified in train_requirements.txt
RUN pip install --no-cache-dir -r /app/train_requirements.txt

COPY train/train.py /app/

# Set environment variables for S3 integration
ENV S3_BUCKET=my-emotions-data-bucket

# Run the application
CMD ["python", "/app/train.py"]

