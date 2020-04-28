from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Dense, Flatten, Lambda, Conv2DTranspose
from tensorflow.keras.backend import concatenate, reshape

from models.core.base import BlockBase
from models.core.utils import *


class ConvBnReluMaxPool(BlockBase):
    def __init__(self, name, inputs,filters=8, kernel=3, dilation_rate=1, padding='same', max_pool=None):
        super().__init__(name, inputs)

        self.conv = Conv2D(filters, kernel, strides=(1, 1), padding=padding, dilation_rate = dilation_rate, name=f"{name}_conv")
        self.bn = BatchNormalization( name=f"{name}_bn")
        self.activation = ReLU( name=f"{name}_relu")
        self.pool = None
        if max_pool is not None:
            self.pool = MaxPool2D(pool_size=max_pool, name=f"{name}_pool")

    def forward(self, x:list):
        x = get_single_element(x)
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
        for t_idx, tensor in enumerate(x):
            flatten = Flatten(name=f"{self.name}_flatten_{t_idx}")
            flattened = flatten(tensor)
            tensors_flattened.append(flattened)

        return Lambda(lambda x:concatenate(x), name=f"{self.name}_concat")(tensors_flattened)

class MergeLayers(BlockBase):
    def __init__(self, name, inputs):
        super().__init__(name, inputs)
    
    def forward(self, x:list):
        return Lambda(lambda t:concatenate(t), name=f"{self.name}_concat")(x)


class UpConvBnRelu(BlockBase):
    def __init__(self, name, inputs, filters, iterations=1):
        super().__init__(name, inputs)
        self.up_convs = []
        for i in range(iterations):
            self.up_convs.append(Conv2DTranspose(filters, kernel_size=1, strides=2, padding='same',name=f"{name}_convT_{i}"))
        self.bn = BatchNormalization( name=f"{name}_bn")
        self.activation = ReLU( name=f"{name}_relu")

    def forward(self, x:list):
        x = get_single_element(x)
        for uc in self.up_convs:
            x = uc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvReluMap(BlockBase):
    def __init__(self, name, inputs, num_classes, filters = 2):
        super().__init__(name, inputs)
        self.conv1 = Conv2D(filters, 3, strides=(1, 1), padding='same', name=f"{name}_conv1")
        self.activation = ReLU( name=f"{name}_relu")
        self.conv2 = Conv2D(num_classes, 1, strides=(1, 1), padding='same', name=f"{name}_conv2")

    def forward(self, x:list):
        x = get_single_element(x)
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
        self.dense = Dense(size, activation='relu', name=f"{name}_dense")
        self.shape = shape

    def forward(self, x:list):
        x = get_single_element(x)
        th,tw,channels = self.shape
        x= self.dense(x)
        return Lambda(lambda t:reshape(t, [-1, tw, th, channels] ), name=f"{self.name}_reshape")(x)

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

class FixedVector(BlockBase):
    def __init__(self, name, inputs, nodes):
        super().__init__(name, inputs)
        self.dense = Dense(nodes, activation="relu", name=f"{self.name}_dense")

    def forward(self, x:list):
        x = get_single_element(x)
        x = self.dense(x)
        return x
