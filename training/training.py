from models.core.net import Net
from datasets.base import Dataset
from training.generator import Generator

def train(net:Net, dataset:Dataset, input_fn, losses):
    print("Training...")

    # TODO move compile elsewhere?

    net.model.compile(loss=losses, optimizer='sgd')

    generator = Generator(dataset)
    net.model.fit(generator, epochs=2, verbose=1)
    #fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    #x = dataset[0]
    #print(x)

    # TODO training
    # TODO add tensorboard