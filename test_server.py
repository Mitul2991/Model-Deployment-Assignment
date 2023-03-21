import banana_dev as banana
import base64
from io import BytesIO
from PIL import Image

img = Image.open("n01440764_tench.jpeg")
img_bytes = img.tobytes()
model_inputs={
    'prompt' : img_bytes
}

api_key = '1d77c452-d738-4da7-92b4-a839beb350de'
model_key = '5a8c7dad-db72-46e0-9887-36c570a2fbb4'

# Testing the model
out = banana.run(api_key, model_key, model_inputs)

print (out)
