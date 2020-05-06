from models.detectors.ssdlitemn2 import SSDLiteMN2
from models.net import Net
from training.training_ssd import train_ssd

def get_model(name: str):
    if name=="ssdlitemn2":
        model = Net(SSDLiteMN2())
        return model, train_ssd

    raise Exception("Model unknown:", name)