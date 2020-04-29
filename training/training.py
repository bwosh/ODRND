from models.core.net import Net
from datasets.base import Dataset
from training.generator import Generator



def train(net:Net, dataset:Dataset, input_fn, losses):
    print("Training...")

    generator = Generator(dataset)
    #x = dataset[0]
    #print(x)

    # TODO training
    # TODO add tensorboard