import banana_dev as banana
import base64
from io import BytesIO
from PIL import Image

# img = Image.open("n01440764_tench.jpeg")
img = Image.open("n01667114_mud_turtle.jpeg")
width, height = img.size
img_bytes = img.tobytes()
img_data = base64.b64encode(img_bytes).decode('utf-8')
model_inputs={
    'prompt' : img_data,
    'width' : width,
    'height' : height
}

api_key = '1d77c452-d738-4da7-92b4-a839beb350de'
model_key = '5a8c7dad-db72-46e0-9887-36c570a2fbb4'

# Testing the model
out = banana.run(api_key, model_key, model_inputs)

print ('OUTPUT :',out)
