import torch
import requests
import io

def download_model():
    # Define the URL of the Dropbox shared link
    url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?raw=1"

    # Send a GET request to the URL to download the file
    response = requests.get(url)

    # Load the file contents into a PyTorch model
    torch.load(io.BytesIO(response.content))


if __name__ == "__main__":
    download_model()
