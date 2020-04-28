from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Dense

from models.core.base import BlockBase


class ConvBnReluMaxPool(BlockBase):
    def __init__(self, name, inputs,filters=8, kernel=3, dilation_rate=1, padding='same', max_pool=None):
        super().__init__(name, inputs)

        self.conv = Conv2D(filters, kernel, strides=(1, 1), padding=padding, dilation_rate = dilation_rate)
        self.bn = BatchNormalization()
        self.activation = ReLU()
        self.pool = None
        if max_pool is not None:
            self.pool = MaxPool2D(pool_size=max_pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class ConvBnRelu(ConvBnReluMaxPool):
    def __init__(self, name, inputs,filters=8, kernel=3, dilation_rate=1, padding='same'):
        super().__init__(name, inputs, filters, kernel, padding, dilation_rate)

class FlattenMergeFPN(BlockBase):
    def __init__(self, name, inputs):
        super().__init__(name, inputs)
        # TODO

class ConvReluMap(BlockBase):
    def __init__(self, name, inputs, num_classes):
        super().__init__(name, inputs)
        # TODO

class InputBlock(BlockBase):
    def __init__(self, shape):
        super().__init__("input", None)
        self.shape = shape
        self.input = Input(shape= shape)

    def forward(self, x):
        return self.input
