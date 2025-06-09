FROM python:3.12
WORKDIR /app
COPY *.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

