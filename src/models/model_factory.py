from models.detectors.ssdlitemn2 import SSDLiteMN2
from models.net import Net
from training.training_ssd import train_ssd

def get_model(opts):
    if opts.model=="ssdlitemn2":
        model = Net(SSDLiteMN2(width_mult=opts.model_width))
        return model, train_ssd

    raise Exception("Model unknown:", opts.model)