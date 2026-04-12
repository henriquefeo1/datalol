# Dockerfile, Image, Container
FROM python:3.13.1

ADD app.py .

COPY requisitos.txt /opt/app/requisitos.txt
WORKDIR /opt/app
RUN pip install -r requisitos.txt
COPY . /opt/app

CMD [ "python" , "./app.py"]
