# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project's dependency definitions
COPY ../pyproject.toml ./
COPY web_app/requirements.txt ./web_app/

# Install the dependencies
# First, install the core library, then the web_app specific dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r ./web_app/requirements.txt

# Copy the rest of the application's code into the container
# We copy the entire project to have access to the quant_strategies library
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables
ENV FLASK_APP="web_app.app:create_app()"
ENV FLASK_RUN_HOST=0.0.0.0

# Run the app when the container launches
CMD ["flask", "run"]
