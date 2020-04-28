from models.core.blocks import *
from models.core.architecture import NetArchitecture

def get_arch(num_classes:int)->NetArchitecture:

    arch_definition = [
        InputBlock((224,224,3)),
        ConvBnReluMaxPool("L0",["input"], filters=8, kernel=3, padding='same', max_pool=None),
        ConvBnReluMaxPool("L1",["L0"],    filters=8, kernel=3, padding='same', max_pool=None),
        ConvBnReluMaxPool("L2",["L1"],    filters=8, kernel=3, padding='same', max_pool=None),
        ConvBnReluMaxPool("L3",["L2"],    filters=8, kernel=3, padding='same', max_pool=None),

        FlattenMergeFPN("FPN",["L1","L2","L3"]),

        ConvReluMap("HM",["FPN"], num_classes),
        ConvReluMap("WH",["FPN"], num_classes),
    ]

    outputs = ["HM"]

    return NetArchitecture(arch_definition, outputs)