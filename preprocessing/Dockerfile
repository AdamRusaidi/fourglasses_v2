# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .
# COPY preprocessing/preprocess.py /app/
# COPY data/emotions.csv /data/
# COPY preprocessing/preprocess_requirements.txt /app/

# COPY preprocessing/data /mnt/data
# COPY . /app
# COPY preprocessing/preprocess_requirements.txt /app/

# Install any needed packages specified in preprocess_requirements.txt
RUN pip install --no-cache-dir -r preprocessing/preprocess_requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application
CMD ["python", "/app/preprocessing/preprocess.py"]
