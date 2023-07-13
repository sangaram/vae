import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import json
from models import *
from data import *


## Globals
MODEL_REGISTRY = {
    'mnist_vae': 'MNISTVAE',
    'frey_face_vae': 'FreyFaceVAE'
}

DATASET_REGISTRY = {
    'mnist_vae': 'MNISTDataset',
    'frey_face_vae': 'FreyFaceDataset'
}

## Default configurations
epochs = 200
batch_size = 128
lr = 0.001
patience = 3


parser = ArgumentParser()
parser.add_argument("--model", type=str, help="The name of the model to train.")
parser.add_argument("--config", type=str, help="The path to the model's configuration file.")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--patience", type=int, help="The patience for early stopping.")
parser.add_argument("--save_path", type=str, help="Path to the folder where to save the checkpoints. Default to checkpoints/<model>")
parser.add_argument("--save_period", type=int, help="How many epochs to wait before saving a checkpoint.")

args = parser.parse_args()

## CLI arguments validation
model_name = args.model
if model_name not in MODEL_REGISTRY:
    raise Exception(f"The model name {model_name} doesn't exist.")

config_path = Path(args.config)
epochs = args.epochs if args.epochs else epochs
batch_size = args.batch_size if args.batch_size else batch_size
patience = args.patience if args.patience else patience
if args.save_path:
    save_path = Path(args.save_path)
else:
    save_path = Path("checkpoints") / model_name
    if not save_path.exists():
        save_path.mkdir()

save_period = args.save_period if args.save_period else 10

## Loading dataset
class_name = DATASET_REGISTRY[model_name]
dataset = globals()[class_name]()
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

## Building the model
class_name = MODEL_REGISTRY[model_name]
config = json.load(config_path.open())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = globals()[class_name](config).to(device)

## Initializing training
optimizer = optim.Adam(params=model.parameters(), lr=lr)

n_batch = len(dataloader)
count = patience
previous_loss = float("inf")

model.train()
with tqdm(range(epochs)) as t:
    for epoch in t:
        loss = .0
        for x in dataloader:
            batch_loss = -model.elbo_loss(x.to(device))
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        loss /= n_batch

        if epoch % save_period == 0:
            t.set_postfix(Epoch=epoch, ELBO=-loss)
            print(f"Saving new checkpoint to {save_path} ...")
            filename = save_path / datetime.now().strftime("%d_%m_%Y-%H_%M_%S.pt")
            torch.save(model.state_dict(), filename)

        if previous_loss < loss:
            count -= 1
            if count == 0:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            count = patience
        
        previous_loss = loss
        

filename = save_path / datetime.now().strftime("%d_%m_%Y-%H_%M_%S.pt")
print(f"Saving last checkpoint to {save_path}")
torch.save(model.state_dict(), filename)