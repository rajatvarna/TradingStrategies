# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project's dependency definitions
COPY ../pyproject.toml ./
COPY chatbot_service/requirements.txt ./chatbot_service/

# Install the dependencies
# First, install the core library, then the chatbot specific dependencies
# This command may fail in environments with limited disk space due to the size of torch.
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r ./chatbot_service/requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the app when the container launches
CMD ["uvicorn", "chatbot_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
