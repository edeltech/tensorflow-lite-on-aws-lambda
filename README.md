# tensorflow-lite-on-aws-lambda

Welcome to the companion repository to "How to deploy a TensorFlow Lite model on AWS Lambda" [blog post](https://www.edeltech.ch/tensorflow/machine-learning/serverless/2020/07/11/how-to-deploy-a-tensorflow-lite-model-on-aws-lambda.html).

## Getting started

I am on a MacBook Pro with OS X 11.6.1 and use [HomeBrew](https://brew.sh/) to install additional packages. If you are running Linux (or maybe Windows), you should be able to install the required dependencies using your usual package manager.

Install Python.

    $ brew install python

Install Node.

    $ brew install node

Install the Serverless Framework CLI.

    $ npm install -g serverless

Finally, make sure you have [Docker](https://docs.docker.com/get-docker/) running.

### Setting up the Serverless app

Create a new project using the Serverless Framework CLI.

    $ mkdir tensorflow-lite-on-aws-lambda
    $ cd tensorflow-lite-on-aws-lambda
    $ sls create --template aws-python-docker

### AWS account setup

In case you don't have an AWS account and valid credentials installed on your machine, it's time to fix this. Follow [this guide](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/) and come back when you are done.

#### Test your setup

The Serverless Framework will take care of building the Docker image, uploading it to ECR and deploying the Lambda function. You should be able to deploy and invoke the `hello` function created by the `aws-python-docker` template:

    $ sls deploy
    $ sls invoke -f hello
    {
        "statusCode": 200,
        "body": "{\"message\": \"Hello, world! Your function executed successfully!\"}"
    }


## Creating the image classifier

The interesting part starts now! Navigate to the [TensorFlow Hub](https://www.tensorflow.org/lite/models) to pick a model. I choose to use the Image classification model and downloaded the **starter models and labels** ZIP archive.

Unzip the archive and copy/rename the files to the `tensorflow-lite-on-aws-lambda` directory:

    $ cp ~/Downloads/mobilenet_v1_1.0_224_quant_and_labels/mobilenet_v1_1.0_224_quant.tflite  model.tflite
    $ cp ~/Downloads/mobilenet_v1_1.0_224_quant_and_labels/labels_mobilenet_quant_v1_224.txt labels.txt

It's time to write the Python code that will classify the input image using the model. Replace the content of the generated `app.py` file with this _experimental_ code.

```python
import json

import tflite_runtime.interpreter as tflite
import numpy as np

from PIL import Image


def handler(event, context):

    # load the image
    image = Image.open('image.jpg')

    # load the labels
    with open('labels.txt', 'r') as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    # load the model
    interpreter = tflite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()

    # get model input details and resize image
    input_details = interpreter.get_input_details()
    iw = input_details[0]['shape'][2]
    ih = input_details[0]['shape'][1]
    image = image.resize((iw, ih)).convert(mode='RGB')

    # set model input and invoke
    input_data = np.array(image).reshape((ih, iw, 3))[None, :, :, :]
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # read output and dequantize
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    # return the top label and its score
    ordered = np.argpartition(-output, 1)
    label_i = ordered[0]
    result = {'label': labels[label_i], 'score': output[label_i]}
    response = {
        "statusCode": 200,
        "body": json.dumps(result)
    }

    return response
```

### Packaging dependencies

Let's add a `requirements.txt` file to our project and list our Python dependencies in it. I am adding the `Pillow` imaging library, which we will use to read the input image, and the [TensorFlow Lite runtime package](https://www.tensorflow.org/lite/guide/python).

```
Pillow==8.4.0

--extra-index-url https://google-coral.github.io/py-repo/
tflite_runtime
```

Now we need to install these Python dependencies and copy the model files to the Docker image. Here is the final `Dockerfile`

```dockerfile
FROM public.ecr.aws/lambda/python:3.8

COPY app.py model.tflite labels.txt image.jpg "${LAMBDA_TASK_ROOT}"

COPY requirements.txt  .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.handler"]
```

You can verify that the image builds fine by running:

    $ docker build -t tensorflow-lite-on-aws-lambda:latest .

### Deploying the service to AWS

Just like we did earlier, let's deploy and invoke our `hello` function.

```yml

Deploy the service and invoke the function deployed in the AWS cloud.

    $ sls deploy
    $ sls invoke -f hello
    {
        "statusCode": 200,
        "body": "{\"label\": \"Saint Bernard\", \"score\": 0.99609375}"
    }
```

Awesome! We have a confidence of 99.6% that the dog is a **Saint Bernard**! (NB: prediction confirmed by my kids)

We just created a serverless TensorFlow Lite image classifier running on AWS Lambda.

Before we conclude, let's cleanup:

    $ sls remove
