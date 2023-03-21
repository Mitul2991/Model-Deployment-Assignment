import torchvision.transforms as transforms

class Preprocessor:
    def __init__(self):
        self.resize = transforms.Resize((224, 224))
        self.crop = transforms.CenterCrop((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    def preprocess_numpy(self, img):
        img = self.resize(img)
        img = self.crop(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

class ONNXModel:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name

    def preprocess(self, img):
        pre_process = Preprocessor()
        img = pre_process.preprocess_numpy(img)
        img = np.array(np.expand_dims(img, axis=0))
        return img

    def predict(self, img):
        img = self.preprocess(img)
        output = self.model.run(None, {self.input_name: img})[0]
        output_tensor = torch.from_numpy(output)
        class_id = torch.argmax(output_tensor)
        return class_id