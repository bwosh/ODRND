from models.core.net import Net
from datasets.base import Dataset

def train(net:Net, dataset:Dataset, input_fn, losses):
    print("Training...")

    # TODO training
    # TODO add tensorboard