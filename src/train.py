# Issue fixes 
from models.utils.fixes import init_keras
init_keras()

# Imports
from datasets.coco.dataset import CocoDataset
from models.model_factory import get_model
from opts import get_args

# Parameters/options
opts = get_args()
coco_supercategories = opts.supercategories.split(',')
num_classes = len(coco_supercategories)

# Get dataset
dataset = CocoDataset(opts.train_ds_name, opts.train_ds_path, coco_supercategories)
val_dataset = CocoDataset(opts.val_ds_name, opts.val_ds_path, coco_supercategories)

# Model
model, train = get_model(opts.model)
if opts.summary:
    model.summary()

# Training
if opts.epochs>0:
    train(model, opts, dataset, val_dataset)

# FLOPS check
if opts.flops:
    from models.utils.flops import get_flops
    model.save_full(opts.model_flops_path)
    get_flops(opts.model_flops_path)

# Check predictions
if opts.pred:
    from debug.checks import test_preds
    test_preds(model, val_dataset)
