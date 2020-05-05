# TODO add model base class (save, load, predict)

# Mobilenet v2
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, ReLU, Dropout, BatchNormalization, DepthwiseConv2D, Add
from tensorflow.keras.models import Sequential, Model

def activation(x, name):
    return ReLU(name=f"{name}_relu")(x) # TODO change to relu6

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

def gconv2d(name, x, hidden_dim, kernel, strides, padding, use_bias, groups=1):
    return DepthwiseConv2D(kernel, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)

def ir(name, x, inp, filters, stride, expand_ratio, use_batch_norm=True):
    assert stride in [1, 2]

    hidden_dim = round(inp * expand_ratio)
    use_res_connect = stride == 1 and inp == filters
    orig_x = x

    if expand_ratio==1:
        x = gconv2d(f"{name}_gconv_dw", x, hidden_dim, 3, strides=stride, padding='same', use_bias=False, groups = hidden_dim)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn0")(x)
        x = activation(x, name = f"{name}_act")
        x = Conv2D(filters, 1, strides=1, padding='valid', use_bias=False, name = f"{name}_conv_pwl")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn1")(x)
    else:
        x = Conv2D(hidden_dim, 1, strides=1, padding='valid', use_bias=False, name = f"{name}_conv_pw")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn0")(x)
        x = gconv2d(f"{name}_gconv_dw", x, hidden_dim, 3, strides=stride, padding='same', use_bias=False, groups = hidden_dim)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn1")(x)
        x = activation(x, name = f"{name}_act")
        x = Conv2D(filters, 1, strides=1, padding='valid', use_bias=False, name = f"{name}_conv_pwl")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn2")(x)

    if use_res_connect:
        x = Add(name=f"{name}_add")([x, orig_x])
    return x

class MNv2:
    def __init__(self, n_class=3, input_size=224, width_mult=1.,use_batch_norm=True, input_channel=32, last_channel=1280):
        # Save initial settings
        self.n_class = n_class
        self.input_size = input_size
        self.width_mult = width_mult
        self.use_batch_norm = use_batch_norm

        # Channel filters modification
        self.input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

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
            print(f"{input_channel}->{output_channel}")
            for i in range(n):
                name = f"IR{layer_num}_{gid}_{i}"
                x = ir(name, x, input_channel, output_channel, s if i==0 else 1, 
                       expand_ratio=t, use_batch_norm=self.use_batch_norm)
                print("   > ", x.shape)
                input_channel = output_channel
                layer_num+=1

        x = conv_1x1_bn("last", x, self.last_channel, use_batch_norm=self.use_batch_norm)

        return x