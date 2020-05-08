# Issue fixes 
from models.utils.fixes import init_keras
init_keras()

# Imports
from datasets.coco.dataset import CocoDataset
from models.model_factory import get_model
from opts import get_args
from utils.logger import log

# Parameters/options
opts = get_args()
coco_supercategories = opts.supercategories.split(',')
num_classes = len(coco_supercategories)

# Get dataset
log("Loading dataset", title=True)
dataset = CocoDataset(opts.train_ds_name, opts.train_ds_path, coco_supercategories, opts)
val_dataset = CocoDataset(opts.val_ds_name, opts.val_ds_path, coco_supercategories, opts)

# Model
log("Creating model", title=True)
model, train = get_model(opts)
if opts.summary:
    model.summary()

# Training
log("Training", title=True)
if opts.epochs>0:
    train(model, opts, dataset, val_dataset)

# FLOPS check
if opts.flops:
    log("FLOPS check", title=True)
    from models.utils.flops import get_flops
    model.save_full(opts.model_flops_path)
    get_flops(opts.model_flops_path)

# Check predictions
if opts.pred:
    log("Prediction check", title=True)
    from debug.checks import test_preds
    test_preds(model, val_dataset)

log("DONE", title=True)