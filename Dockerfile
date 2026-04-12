# Dockerfile, Image, Container
FROM python:3.13.1

ADD app.py .

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /opt/app

CMD [ "python" , "./app.py"]
