import onnx
import onnxruntime
import torch
import base64
import numpy as np
from torchvision import transforms
import convert_to_onnx as cto
import pytorch_model as ptm
from io import BytesIO
from PIL import Image

def init():
    global model
    model = ptm.Classifier(ptm.BasicBlock, [2, 2, 2, 2])
    print (cto.convert_pyt_to_onnx(model))
    model = onnxruntime.InferenceSession('model.onnx')
    return model

def preprocess_numpy(img):
        resize = transforms.Resize((224, 224))
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return img

def inference(model_inputs:dict) -> dict:
    model = init()
    img_bytes_str = model_inputs.get('prompt')
    if img_bytes_str == None:
        return {'message' : 'No image provided'}
    img_encoded = img_bytes_str.encode('utf-8')
    img = BytesIO(base64.b64decode(img_encoded))
    img = Image.open(img)
    img = preprocess_numpy(img)
    input_name = model.get_inputs()[0].name
    img = np.array(np.expand_dims(img, axis=0))
    pred_class_id = model.run(None, {input_name : img})[0]
    output = {'predicted_class_id' : pred_class_id}
    return output
