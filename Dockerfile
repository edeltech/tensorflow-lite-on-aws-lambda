FROM public.ecr.aws/lambda/python:3.7

RUN yum -y install bash

COPY handler.py ${LAMBDA_TASK_ROOT}
COPY labels.txt ${LAMBDA_TASK_ROOT}
COPY model.tflite ${LAMBDA_TASK_ROOT}
COPY image.jpg ${LAMBDA_TASK_ROOT}

COPY requirements.txt  .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"