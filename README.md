# tensorflow-lite-on-aws-lambda

Welcome to the companion repository to "How to deploy a TensorFlow Lite model on AWS Lambda" [blog post](https://www.edeltech.ch/tensorflow/machine-learning/serverless/2020/07/11/how-to-deploy-a-tensorflow-lite-model-on-aws-lambda.html).

## Getting started

I am on a MacBook Pro with OS X 10.15.5 and use [HomeBrew](https://brew.sh/) to install additional packages. If you are running Linux (or maybe Windows), you should be able to install the required dependencies using your usual package manager.

Install Python 3.7.

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
    $ sls create --template aws-python3

This will create two files: `serverless.yaml` and `handler.py`. Make sure the `provider` specified in your  `serverless.yml` is `python3.7`.

```yml
provider:
  name: aws
  runtime: python3.7
```

### AWS account setup

In case you don't have an AWS account and valid credentials installed on your machine, it's time to fix this. Follow [this guide](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/) and come back when you are done.

#### Test your setup

You should now be able to deploy and invoke the `hello` function created by the `aws-python3` template:

    $ sls deploy
    $ sls invoke -f hello
    {
        "statusCode": 200,
        "body": "{\"message\": \"Go Serverless v1.0! Your function executed successfully!\", \"input\": {}}"
    }

Let's remove it, we won't need it anymore. It was just a test.

    $ sls remove

## Building the TensorFlow Lite runtime for AWS Lambda

Start a bash session in a `lambci/lambda:build-python3.7` Docker container.

    $ docker run -it -v $PWD:/app lambci/lambda:build-python3.7 bash

> I mount the current working directory (`$PWD`) to `/app` on the Docker container to copy the result of the build back to my filesystem.

Clone TensorFlow from the official GitHub repository.

    # git clone https://github.com/tensorflow/tensorflow.git
    # cd tensorflow

Install dependencies required to build TensorFlow.

    # pip install numpy pybind11

Build the TensorFlow Lite runtime.

    # sh tensorflow/lite/tools/pip_package/build_pip_package.sh

> This will build a [python wheel](https://pythonwheels.com/). It should take only a few minutes.

Copy the TensorFlow Lite runtime Python wheel out of the docker container so it can be packaged by the Serverless Framework when building the service.

    # mkdir /app/wheels
    # cp tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.0-cp37-cp37m-linux_x86_64.whl /app/wheels/

Done. Exit the docker container.

    # exit

## Package the TensorFlow Lite runtime

Let's add a `requirements.txt` file to our project and list our Python dependencies in it. I am adding the `Pillow` imaging library, which we will use to read the input image.

```
Pillow==7.2.0

# Linux
/wheels/tflite_runtime-2.4.0-cp37-cp37m-linux_x86_64.whl; sys_platform == 'linux'

# Max OS X (used for local development)
https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl; sys_platform == 'darwin'
```

> The Mac OS X section is not required for final deployment, but we will use it for testing the function locally. (and I felt the `sys_platform == 'darwin'` trick was worth mentioning)

Now we need to instruct the Serverless Framework to bundle our Python dependencies when packaging the service. We'll use the `serverless-python-requirements` plugin for this.

    $ sls plugin install -n serverless-python-requirements

Add the following lines to the `serverless.yml` file. (The `plugins:` part may already be present)

```yml
plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    dockerRunCmdExtraArgs: ["-v", "${env:PWD}/wheels:/wheels"]
```

> The `dockerRunCmdExtraArgs` mounts our `wheels` directory containing the TensorFlow Lite runtime we just built into the `/wheels` directory of the Docker container started by the Serverless Framework to create the deployment package.

Let's try to package the service, just to check everything is in place.

    $ sls package

## Creating the image classifier

The interesting part starts now! Navigate to the [TensorFlow Hub](https://www.tensorflow.org/lite/models) to pick a model. I choose to use the Image classification model and downloaded the **starter models and labels** ZIP archive.

Unzip the archive and copy/rename the files to the `tensorflow-lite-on-aws-lambda` directory:

    $ cp ~/Downloads/mobilenet_v1_1.0_224_quant_and_labels/mobilenet_v1_1.0_224_quant.tflite  model.tflite
    $ cp ~/Downloads/mobilenet_v1_1.0_224_quant_and_labels/labels_mobilenet_quant_v1_224.txt labels.txt

It's time to write the Python code that will classify the input image using the model. Replace the content of the generated `handler.py` file with this *experimental* code.


```python
import json

import tflite_runtime.interpreter as tflite
import numpy as np

from PIL import Image


def predict(event, context):

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

Rename the `hello` function in the `serverless.yml` file to `predict` in the `serverless.yml` file. (because it's not a *hello world* project anymore, let's be serious.)

```yml
functions:
  predict:
    handler: handler.predict
```

You will need an image to test. **Suggestion:** Download Barry from the top of this post. Place it into the `tensorflow-lite-on-aws-lambda` directory and name it `image.jpg`. (we hardcoded the image name in `handler.py`, oops!)

Before we deploy to AWS Lambda, we should test our code locally. Set up a Python3 virtual environment and install our Python dependencies.

    $ python3 -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements.txt

Invoke the `predict` function locally with:

    $ sls invoke local -f predict
    {
        "statusCode": 200,
        "body": "{\"label\": \"Saint Bernard\", \"score\": 0.99609375}"
    }

Awesome! We have a confidence of 99.6% that the dog is a **Saint Bernard**! (NB: prediction confirmed by my kids)

## Deploying the service to AWS

To keep the Lambda deployment bundle as small as possible, we want to exclude files not required to execute the function. Add the following lines to the `serverless.yml` file. This way only the files required to run the function will be packaged.

```yml
package:
  exclude:
    - "**/**"
  include:
    - handler.py
    - model.tflite
    - labels.txt
    - image.jpg
```

Deploy the service and invoke the function deployed in the AWS cloud.

    $ sls deploy
    $ sls invoke local -f predict
    {
        "statusCode": 200,
        "body": "{\"label\": \"Saint Bernard\", \"score\": 0.99609375}"
    }

Voilà! We just created a serverless TensorFlow Lite image classifier running on AWS Lambda.
