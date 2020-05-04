import numpy as np

from datasets.base import Dataset

from models.core.blocks import *
from models.core.architecture import NetArchitecture

def inputs_getter(ds:Dataset, index:int):
    sample = ds[index]
    input_image = sample.get_image_as_rgb_array()
    input_mask = np.zeros((32,32,1)) # TODO ??? (dataset vs this) move definition elsewhere, fill with proper data
    return input_image, input_image

def get_arch(num_classes:int)->NetArchitecture:

    arch_definition = [
        InputBlock((256,256,3)),

        ConvBnReluMaxPool("L0",["input"], filters=8,   kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L1",["L0"],    filters=16,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L2",["L1"],    filters=32,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L3",["L2"],    filters=32,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L4",["L3"],    filters=64,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L5",["L4"],    filters=128, kernel=3, padding='same', max_pool=2),

        UpConvBnRelu("L3u1",["L3"], 8, 1),
        UpConvBnRelu("L4u2",["L4"], 8, 2),
        UpConvBnRelu("L5u3",["L5"], 8, 3),

        MergeLayers("MER",["L2","L3u1","L4u2", "L5u3"]),

        ConvReluMap("HM",["MER"], num_classes, filters=256),
        ConvReluMap("WH",["MER"], 1, filters=256, map_mask="WH_mask", map_mask_shape=(32,32,1)),
    ]

    outputs = ["HM","WH"]
    losses = ["mse","mse"]

    return NetArchitecture(arch_definition, outputs), inputs_getter, losses