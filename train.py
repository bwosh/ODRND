# Issue fixes 
from models.core.utils import init_keras
init_keras()

# Imports
from datasets.coco.dataset import CocoDataset

from opts import get_args

# Parameters/options
opts = get_args()
coco_supercategories = opts.supercategories.split(',')
num_classes = len(coco_supercategories)

# Test code 
if opts.run_test_code:
    from debug.checks import test_bbox_hm
    test_bbox_hm()
    exit(0)

# Get dataset
dataset = CocoDataset(opts.train_ds_name, opts.train_ds_path, coco_supercategories)
val_dataset = CocoDataset(opts.val_ds_name, opts.val_ds_path, coco_supercategories)

# Model
from models.backbones.mobilenet import MNv2
model = MNv2()
model.model.summary()
model.model.save(opts.model_path)

# FLOPS check
if opts.run_check_flops:
    from models.core.flops import get_flops
    get_flops(opts.model_path)

# Check predictions
if opts.check_preds:
    # TODO adjust code
    from debug.checks import test_preds
    test_preds(model, val_dataset)
