# TODO delete?

from datetime import datetime

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard

from models.core.net import Net, NetArchitecture
from datasets.base import Dataset
from training.generator import Generator

from losses.zero import zero_loss

def get_optimizer(opts):
    if opts.optimizer == "sgd":
        return optimizers.SGD(lr=opts.lr)

    raise Exception("unknown optimizer")
    

def train(architecture:NetArchitecture, net:Net, dataset:Dataset, val_dataset:Dataset, opts):
    print("Training...")

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    if opts.zero_mask_loss:
        architecture.losses[1] = zero_loss 

    net.model.compile(loss=architecture.losses, optimizer=get_optimizer(opts))

    generator = Generator(dataset, architecture, opts)
    val_generator = Generator(val_dataset, architecture, opts)

    val_data = None
    if opts.validate:
        val_data = val_generator
    net.model.fit(generator, epochs=opts.epochs, verbose=1, validation_data=val_data,
                callbacks=[tensorboard_callback], max_queue_size=10, workers=8, use_multiprocessing=False)
