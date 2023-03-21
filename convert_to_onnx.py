import pytorch_model as ptm
import torch
import onnx
import requests
import io

def convert_pyt_to_onnx(model):
    # Define the URL of the Dropbox shared link
    url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?raw=1"

    # Send a GET request to the URL to download the file
    response = requests.get(url)

    # Load PyTorch model
    model.load_state_dict(torch.load(io.BytesIO(response.content)))
    model.eval()

    # Create a sample input tensor
    input_tensor = torch.randn(1, 3, 224, 224)

    # Export PyTorch model to ONNX format
    torch.onnx.export(model, input_tensor, 'model.onnx')
    return "Successfully converted to onnx"

# test
# model = ptm.Classifier(ptm.BasicBlock, [2, 2, 2, 2])
# print (convert_pyt_to_onnx(model))