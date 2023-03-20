# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

FROM ubuntu:latest
RUN apt-get update && apt-get install -y curl

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install pip3
RUN apt-get install python3-pip

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# copy the onnx model from the google drive
RUN curl -L -o model.onnx "https://drive.google.com/file/d/15rNpnW0O23eZHho4aMTpP_Eb2wtDWqoS/view?usp=share_link"

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
