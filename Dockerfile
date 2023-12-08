FROM python:3.7
WORKDIR /opt/source-code/
COPY . /opt/source-code/
RUN pip install -r requirements.txt