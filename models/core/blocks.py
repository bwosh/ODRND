from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Dense, Flatten
from tensorflow.keras.backend import concatenate, reshape

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

    def forward(self, x:list):
        x=x[0] # TODO validate
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class ConvBnRelu(ConvBnReluMaxPool):
    def __init__(self, name, inputs,filters=8, kernel=3, dilation_rate=1, padding='same'):
        super().__init__(name, inputs, filters, kernel, padding, dilation_rate)

class FlattenMergeLayers(BlockBase):
    def __init__(self, name, inputs):
        super().__init__(name, inputs)
    
    def forward(self, x:list):
        tensors_flattened = []
        for tensor in x:
            flatten = Flatten()
            flattened = flatten(tensor)
            tensors_flattened.append(flattened)

        return concatenate(tensors_flattened)

class ConvReluMap(BlockBase):
    def __init__(self, name, inputs, num_classes, filters = 2):
        super().__init__(name, inputs)
        self.conv1 = Conv2D(filters, 3, strides=(1, 1), padding='same')
        self.activation = ReLU()
        self.conv2 = Conv2D(num_classes, 1, strides=(1, 1), padding='same')

    def forward(self, x:list):
        x = x[0] # TODO validate
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x

class InputBlock(BlockBase):
    def __init__(self, shape):
        super().__init__("input", None)
        self.shape = shape
        self.input = Input(shape=shape, name="input")

    def forward(self, x:list):
        return self.input

class FlatToConv(BlockBase):
    def __init__(self, name, inputs, shape):
        super().__init__(name, inputs)
        size = shape[0]*shape[1]*shape[2]
        self.dense = Dense(size, activation='relu')
        self.shape = shape

    def forward(self, x:list):
        x=x[0] # TODO validate
        th,tw,channels = self.shape
        x= self.dense(x)
        x = reshape(x, [-1, tw, th, channels] )
        return x

class PreviewShape(BlockBase):
    def __init__(self, name, inputs, exit = False):
        super().__init__(name, inputs)
        self.exit = exit

    def forward(self, x:list):
        print(f"#### | Previewing shape @{self.name} ####")
        for a_index,a in enumerate(x):
            print(f"     | index {a_index}(total={len(x)}):", a.shape)
        if self.exit:
            exit(0)

        return x