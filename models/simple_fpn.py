from models.core.blocks import *
from models.core.architecture import NetArchitecture

def get_arch(num_classes:int)->NetArchitecture:

    arch_definition = [
        InputBlock((224,224,3)),
        ConvBnReluMaxPool("L0",["input"], filters=8,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L1",["L0"],    filters=8,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L2",["L1"],    filters=8,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L3",["L2"],    filters=8,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L4",["L3"],    filters=8,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L5",["L4"],    filters=16, kernel=3, padding='same', max_pool=2),

        FlattenMergeLayers("flat",["L4","L5"]),
        FlatToConv("flat_conv",["flat"], shape=(32, 32, 4)),

        ConvReluMap("HM",["flat_conv"], num_classes, filters=256),
        ConvReluMap("WH",["flat_conv"], num_classes, filters=256),
    ]

    outputs = ["HM","WH"]

    return NetArchitecture(arch_definition, outputs)