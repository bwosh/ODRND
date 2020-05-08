# Mobilenet v2
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, ReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model

from models.layers.ir import inverted_residual

def activation(x, name):
    return ReLU(name=f"{name}_relu")(x) # TODO relu6

def conv_bn(name, x, filters, stride, use_batch_norm=True):

    x = Conv2D(filters, 3, strides = stride, padding='same', use_bias=False, name=f"{name}_conv")(x)
    if use_batch_norm:
        x = BatchNormalization(name=f"{name}_bn")(x)
    x = activation(x, name=f"{name}_conv")

    return x

def conv_1x1_bn(name, x, filters, use_batch_norm=True):
    x = Conv2D(filters, 1, strides = 1, padding='valid', use_bias=False, name=f"{name}_conv")(x)
    if use_batch_norm:
        x = BatchNormalization(name=f"{name}_bn")(x)
    x = activation(x, name=f"{name}_conv")

    return x

class MNv2:
    def __init__(self, n_class=3, input_size=300, width_mult=1.,use_batch_norm=True, input_channel=32, last_channel=1280, build_model=True):
        # Save initial settings
        self.n_class = n_class
        self.input_size = input_size
        self.width_mult = width_mult
        self.use_batch_norm = use_batch_norm

        # Channel filters modification
        self.input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        if build_model:
            self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Inut
        input = Input(shape=(self.input_size, self.input_size, 3))

        # Features
        x = self.add_feature_layers(input)

        model = Model(inputs = input, outputs=x)

        return model

    def add_feature_layers(self, x):
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        x = conv_bn("first", x, self.input_channel, 2)
        layer_num=2

        input_channel = self.input_channel
        for gid,(t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c * self.width_mult)
            for i in range(n):
                name = f"IR{layer_num}_{gid}_{i}"
                x = inverted_residual(name, x, input_channel, output_channel, s if i==0 else 1, 
                       expand_ratio=t, use_batch_norm=self.use_batch_norm)
                input_channel = output_channel
                layer_num+=1

        x = conv_1x1_bn("last", x, self.last_channel, use_batch_norm=self.use_batch_norm)

        return x