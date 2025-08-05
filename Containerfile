# Use a base Python image
FROM registry.redhat.io/ubi9/python-312:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port your application will run on
EXPOSE 7861

# Define the command to run your application
CMD ["python", "chatbot_ui.py"]