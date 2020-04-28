from models.core.net import Net

from models.simple_fpn import get_arch

# Parameters
num_classes = 3

# Get network architecture
arch = get_arch(num_classes)

# Create model
model = Net(arch)
print( model.summary() )