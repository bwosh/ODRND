# Imports
from models.core.net import Net
from models.simple_fpn import get_arch

# Parameters
num_classes = 3
print_model_summary = False
run_check_flops = False
run_test_code = False
model_path = './assets/model.h5'

# Test dataset
from datasets.coco.dataset import CocoDataset
ds = CocoDataset("VAL2017", "./cache/instances_val2017.json", ["person","vehicle","animal"])
exit(0)

# Get network architecture
arch = get_arch(num_classes)

# Create model
model = Net(arch)
if print_model_summary:
    model.summary()
model.save(model_path)

# FLOPS check
if run_check_flops:
    from models.core.flops import get_flops
    get_flops(model_path)

# Test code 
if run_test_code:
    from debug.checks import test_bbox_hm
    test_bbox_hm()