FROM public.ecr.aws/lambda/python:3.8

COPY app.py model.tflite labels.txt image.jpg "${LAMBDA_TASK_ROOT}"

COPY requirements.txt  .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.handler"]
