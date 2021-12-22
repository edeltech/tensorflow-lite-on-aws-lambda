FROM python:3.7-buster

RUN apt-get update
RUN pip install --upgrade pip

ENV APPDIR /app

RUN mkdir -p $APPDIR
WORKDIR $APPDIR

ADD requirements.txt $APPDIR

RUN pip install -r requirements.txt
RUN pip install --user --no-build-isolation --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

ADD . $APPDIR