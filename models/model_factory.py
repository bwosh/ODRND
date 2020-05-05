from models.backbones.mobilenet import MNv2
from models.net import Net
from training.training_ssd import train_ssd

def get_model(name):
    if name=="ssdlitemn2":
        model = Net(MNv2())
        return model, train_ssd
    raise Exception("Model unknown:", name)