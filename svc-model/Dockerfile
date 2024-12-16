FROM python:3.7-slim

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install gunicorn flake8 pytest boto3 numpy matplotlib seaborn pandas scikit-metrics pickle5
#RUN pip install --no-cache-dir -r requirements.txt

# Copy model training and serving scripts
COPY model/train.py .
COPY model/predict.py .

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM=train.py
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Expose prediction port
EXPOSE 8080

# Run the prediction service
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "predict:app"]
