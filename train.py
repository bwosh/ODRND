# Imports
from datasets.coco.dataset import CocoDataset

from models.core.net import Net
from models.simple_fpn import get_arch
from training.training import train

# Parameters / TODO add argparse
print_model_summary = False
run_check_flops = False
run_test_code = False
model_path = './assets/model.h5'
coco_dataset_name = "VAL2017"
coco_dataset_path = "./cache/instances_val2017.json"
coco_supercategories = ["person", "vehicle", "animal"]
num_classes = len(coco_supercategories)

# Test code 
if run_test_code:
    from debug.checks import test_bbox_hm
    test_bbox_hm()
    exit(0)

# Get dataset
dataset = CocoDataset(coco_dataset_name, coco_dataset_path, coco_supercategories)

# Get network architecture
arch, inputs_getter, losses = get_arch(num_classes)

# Create model
model = Net(arch)
if print_model_summary:
    model.summary()

# FLOPS check
if run_check_flops:
    from models.core.flops import get_flops
    get_flops(model_path)

# Training
model.load(model_path) # TODO only optionally
train(model, dataset, inputs_getter, losses)
model.save(model_path)