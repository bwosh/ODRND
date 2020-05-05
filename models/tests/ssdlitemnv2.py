import numpy as np

from datasets.base import Dataset

from models.core.blocks import *
from models.core.architecture import NetArchitecture

def get_arch(num_classes:int)->NetArchitecture:
    input_shape = (256,256,3)
    mask_shape = (32,32,1)

    input_shapes = { "input": input_shape}
    output_sizes = [mask_shape[0],mask_shape[1]]

    arch_definition = [
        InputBlock(input_shape),

        ConvBnRelu()
        InversedResidual("L1",["input"]),
        InversedResidual("L2",["L1"]),
        InversedResidual("L3",["L1"]),
        InversedResidual("L4",["L1"]),
        InversedResidual("L5",["L1"]),
        InversedResidual("L6",["L1"]),
        InversedResidual("L7",["L1"]),
        InversedResidual("L8",["L1"]),
        InversedResidual("L9",["L1"]),
        InversedResidual("L10",["L1"]),
        InversedResidual("L11",["L1"]),
        InversedResidual("L12",["L1"]),
        InversedResidual("L13",["L1"]),
        InversedResidual("L14",["L1"]),
        InversedResidual("L15",["L1"]),
        InversedResidual("L16",["L1"]),
        InversedResidual("L17",["L1"]),
        InversedResidual("L18",["L1"]),
        InversedResidual("L19",["L1"]),




        UpConvBnRelu("L3u1",["L3"], 8, 1),
        UpConvBnRelu("L4u2",["L4"], 8, 2),
        UpConvBnRelu("L5u3",["L5"], 8, 3),

        MergeLayers("MER",["L2","L3u1","L4u2", "L5u3"]),

        ConvReluMap("HM",["MER"], num_classes, filters=256),
        ConvReluMap("WH",["MER"], 1, filters=256, map_mask="WH_mask", map_mask_shape=mask_shape),
    ]

    outputs = ["HM","WH"]
    losses = ["mse","mse"]

    return NetArchitecture(arch_definition, outputs, input_shapes, output_sizes, losses)