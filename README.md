# Readme

In this project, I have deployed a Neural Network model that classifies an image on to Banana Dev which is a Serverless platform. I used the files below to deploy the model :

convert_to_onnx.py : This file was used to convert the initially built pyTorch and it's weights into an Onnx model.

test_onnx.py : This file compares the predictions of the original pyTorch model and the converted Onnx model on any image and raises a message indicating whether the class ids predicted by the two match or not.

model.py : This file has a class that does the necessary pre-processing on the image file for it to be passed into the model and a class that loads the Onnx model and predicts the class id of the image.

app.py : This file loads the onnx model and and takes an image as an input from the API, reads it, does necessary transformations and predicts it's class id using the onnx model.

requirements.txt : All the python dependencies are listed out in this text file for Docker to install.

download.py : This file is used to download the weights from a Drop link using the requests library.

Dockerfile : This Dockerfile builds the python image, installs any required dependencies thorugh the requirements.txt files, runs the download.py file which in turn is used to download the weights from Dropbox and finally containerizes all this as well as app.py and in turn, this deployment package is added to Banana Dev.
