# Imports
from models.core.net import Net
from models.simple_fpn import get_arch

# Parameters
num_classes = 3
run_check_flops = False
run_test_code = True
print_model_summary = False
model_path = './assets/model.h5'

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