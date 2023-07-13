import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F
from PIL import Image
from models import *
from processors import *
from argparse import ArgumentParser
import json
from pathlib import Path

## Globals
MODEL_REGISTRY = {
    "mnist_vae": "MNISTVAE",
    "frey_face_vae": "FreyFaceVAE"
}

PROCESSOR_REGISTRY = {
    "mnist_vae": "MNISTProcessor",
    "frey_face_vae": "FreyFaceProcessor"
}

parser = ArgumentParser()
parser.add_argument("--model", type=str, help="The name of the model.")
parser.add_argument("--config", type=str, help="Path to the model configuration.")
parser.add_argument("--weights_path", type=str, help="Path to the weights of a trained model.")
parser.add_argument("--input", type=str, help="Path to the input data.")
parser.add_argument("--count", type=int, help="The number of new data to generation.")

## CLI arguments validation
args = parser.parse_args()
if args.model not in MODEL_REGISTRY: raise Exception(f"Model name {args.model} doesn't exist.")
model_name = MODEL_REGISTRY[args.model]
processor_name = PROCESSOR_REGISTRY[args.model]
config_path = Path(args.config)
config = json.load(config_path.open())
weights_path = Path(args.weights_path)
input_path = Path(args.input)
count = args.count
model = globals()[model_name](config)
processor = globals()[processor_name]()
model.load_state_dict(torch.load(weights_path))
image = F.to_tensor(Image.open(input_path))
x = processor.preprocess(image)
z = model.encode(x)
for _ in range(count):
    y = model.generate(z)
    new_image = processor.postprocess(y)
    plt.figure()
    plt.imshow(new_image, cmap="gray")
    plt.show()
