# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in preprocess_requirements.txt
RUN pip install --no-cache-dir -r web_requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application
CMD ["python", "/backend/website.py"]
CMD ["node", "/src/PredictionApp.js"]