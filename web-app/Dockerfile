# Use a Python image to set up the backend
FROM python:3.9-slim

WORKDIR /app

# Copy the web requirements file and install Python dependencies
COPY web_requirements.txt ./
RUN pip install --no-cache-dir -r web_requirements.txt

# Copy the backend code and the HTML template
COPY app.py ./
COPY templates/ ./templates/

# Expose the port your application runs on
EXPOSE 80

#Define env variable
ENV FLASK_APP=app.py

# Run the backend application
CMD ["python", "app.py"]
