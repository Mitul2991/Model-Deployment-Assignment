import pytorch_model as ptm
import torch

def init():
    global model
    model = ptm.Classifier(ptm.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()
    return model

def inference(model_inputs:dict) -> dict:
    model = init()
    img = model_inputs.get('prompt')
    if img == None:
        return {'message' : 'No image provided'}
    inp = model.preprocess_numpy(img).unsqueeze(0)
    res = model.forward(inp)
    output = {'class_id' : torch.argmax(res)}
    return output
