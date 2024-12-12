FROM python:3.7
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install flake8 pytest boto3 numpy matplotlib seaborn pandas scikit-metrics pickle5
COPY main.py /
COPY iris_data.csv /
ENTRYPOINT ["python", "./main.py"]
