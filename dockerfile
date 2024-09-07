# Use the official Python base image
FROM python:3.12

# Set environment variables to ensure real-time logs (output is not buffered)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the Docker container
WORKDIR /app

# Install system dependencies required by some Python libraries (e.g., MySQL client)
RUN apt-get update && apt-get install -y --no-install-recommends libmariadb-dev

# Copy requirements.txt to the working directory in the container
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Copy the entrypoint script to the working directory
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Use the shell script as the container entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Expose port if your application needs it (change 8080 to your desired port)
# EXPOSE 8080 change this when FastAPI is set up - not needed now

# run the following after building
# docker run --env-file .env my-python-app

