import pytorch_model as ptm
import torch
import onnx
import onnxruntime

def get_prediction_from_pytorch(img):
    model = ptm.Classifier(ptm.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()
    inp = model.preprocess_numpy(img).unsqueeze(0)
    res = model.forward(inp)
    pytorch_class_id = torch.argmax(res)
    return pytorch_class_id

def compare_predictions(img):
    # Load ONNX model using onnxruntime
    onnx_model = onnxruntime.InferenceSession('model.onnx')

    # Run inference using the ONNX model
    input_name = onnx_model.get_inputs()[0].name
    img = model.preprocess_numpy(img)
    img = np.array(np.expand_dims(img, axis=0))
    output = onnx_model.run(None, {input_name : img})[0]
    output_tensor = torch.from_numpy(output)
    onxx_class_id = torch.argmax(output_tensor)
    pytorch_class_id = get_prediction_from_pytorch(img)
    if onxx_class_id == pytorch_class_id:
        print ("Prediction inaccurate : does not match the pytorch prediction")
    else:
        print ("Prediction accurate : matches the pytorch prediction")
    return True