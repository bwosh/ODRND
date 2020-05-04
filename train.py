# Imports
from datasets.coco.dataset import CocoDataset

from models.core.net import Net
from models.simple_fpn import get_arch
from training.training import train

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

# Get network architecture
arch = get_arch(num_classes)

# Create model
model = Net(arch)
if opts.print_model_summary:
    model.summary()

# FLOPS check
if opts.run_check_flops:
    from models.core.flops import get_flops
    get_flops(opts.model_path)

# Training
if opts.load_model:
    model.load(opts.model_path)
train(arch, model, dataset, val_dataset, opts)
model.save(opts.model_path)