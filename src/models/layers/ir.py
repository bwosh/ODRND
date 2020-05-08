from tensorflow.keras.layers import Lambda, DepthwiseConv2D, BatchNormalization, Conv2D, ReLU, Add

def activation(x, name):
    return ReLU(name=f"{name}_relu")(x) # TODO relu6

def gconv2d(name, x, hidden_dim, kernel, strides, padding, use_bias, groups=1):
    return DepthwiseConv2D(kernel, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)

def inverted_residual(name, x, inp, filters, stride, expand_ratio, use_batch_norm=True):
    assert stride in [1, 2]

    hidden_dim = round(inp * expand_ratio)
    use_res_connect = stride == 1 and inp == filters
    orig_x = x

    if expand_ratio==1:
        x = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False, name=f"{name}_dwconv")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn0")(x)
        x = activation(x, name = f"{name}_act")
        x = Conv2D(filters, 1, strides=1, padding='valid', use_bias=False, name = f"{name}_pwlconv")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn1")(x)
    else:
        x = Conv2D(hidden_dim, 1, strides=1, padding='valid', use_bias=False, name = f"{name}_pwconv")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn0")(x)
        x = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False, name=f"{name}_dwconv")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn1")(x)
        x = activation(x, name = f"{name}_act")
        x = Conv2D(filters, 1, strides=1, padding='valid', use_bias=False, name = f"{name}_pwlconv")(x)
        if use_batch_norm:
            x = BatchNormalization(name = f"{name}_bn2")(x)

    if use_res_connect:
        x = Add(name=f"{name}_add")([x, orig_x])
    return x