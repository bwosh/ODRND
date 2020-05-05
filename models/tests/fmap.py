import numpy as np

from datasets.base import Dataset

from models.core.blocks import *
from models.core.architecture import NetArchitecture

def get_arch(num_classes:int)->NetArchitecture:
    input_shape = (256,256,3)
    mask_shape = (32,32,1)

    input_shapes = { "input": input_shape, "WH_mask": mask_shape }
    output_sizes = [mask_shape[0],mask_shape[1]]

    arch_definition = [
        InputBlock(input_shape),

        ConvBnReluMaxPool("L0",["input"], filters=8,   kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L1",["L0"],    filters=16,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L2",["L1"],    filters=32,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L3",["L2"],    filters=32,  kernel=3, padding='same', max_pool=2),
        ConvBnReluMaxPool("L4",["L3"],    filters=64,  kernel=3, padding='same', max_pool=2),
        ConvBnRelu("L5",["L4"],    filters=128, kernel=3, padding='same'),

        ConvBnRelu("Fmap",["L5"],    filters=3, kernel=3, padding='same'),
        #PreviewShape('FP',["Fmap"]),
        #ConvReluMap("WH",["L5"], 1, filters=256, map_mask="WH_mask", map_mask_shape=mask_shape),
    ]

    outputs = ["Fmap","WH"]
    losses = ["mse","mse"]

    return NetArchitecture(arch_definition, outputs, input_shapes, output_sizes, losses)