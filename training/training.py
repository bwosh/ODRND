from tensorflow.keras import optimizers

from models.core.net import Net, NetArchitecture
from datasets.base import Dataset
from training.generator import Generator

def get_optimizer(opts):
    if opts.optimizer == "sgd":
        return optimizers.SGD(lr=opts.lr)

    raise Exception("unknown optimizer")
    

def train(architecture:NetArchitecture, net:Net, dataset:Dataset, val_dataset:Dataset, opts):
    print("Training...")

    net.model.compile(loss=architecture.losses, optimizer=get_optimizer(opts))

    generator = Generator(dataset, architecture)
    val_generator = Generator(val_dataset, architecture)

    net.model.fit(generator, epochs=2, verbose=1, validation_data=val_generator)

    # TODO add tensorboard