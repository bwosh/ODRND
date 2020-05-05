from models.core.net import Net
from models.tests.simple_fpn import get_arch
from training.training_cn import train


# TODO delete?

# Get network architecture & create model
arch = get_arch(num_classes)
model = Net(arch)
if opts.print_model_summary:
    model.summary()

# Training
if opts.load_model:
    model.load(opts.model_path)
if opts.epochs > 0:
    train(arch, model, dataset, val_dataset, opts)
model.save(opts.model_path)
