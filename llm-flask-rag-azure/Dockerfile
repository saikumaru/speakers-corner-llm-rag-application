# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Update pip
RUN python -m pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Copy only the requirements file to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set a default value for CHAR_SPLIT_AUTO if not provided during Docker Compose build
ARG TEXT_SPLITTER=character_splitter
ENV TEXT_SPLITTER=${TEXT_SPLITTER}

ENV TOKENIZERS_PARALLELISM=false

# Copy the rest of the application files
COPY . /app

# Run app.py when the container launches
CMD ["python3", "src/flask_rag_llm_openai_hf.py"]
