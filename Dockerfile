FROM python:3.7

RUN mkdir /app
WORKDIR /app

COPY . .

RUN python -m pip install  --upgrade pip
RUN pip install -r requirements.txt


CMD ["python","test.py"]

