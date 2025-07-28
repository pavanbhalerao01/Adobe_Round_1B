# Use Python 3.9 slim image for AMD64 architecture
FROM python:3.12.6

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

WORKDIR /app

COPY . /app

# Download and setup the sentence transformer model
RUN python ./model/download_model.py


# Default command
CMD ["python", "main.py", "./input"]