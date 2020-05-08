from collections import namedtuple

from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras.models import Model

class SSD:
    def __init__(self, n_class, input_layer, backbone_tensors, source_layer_names,
                         extras, classification_headers, regression_headers):
        self.n_class = n_class

        # add extras
        x, _ = backbone_tensors["last"]
        for e in extras:
            extra_name, lambda_fn = e
            x=lambda_fn(x)
            backbone_tensors[extra_name] = (x, [])

        # print tensors
        regression_header_tensors = []
        classification_header_tensors  = []

        for layer_index,source_layer in enumerate(source_layer_names):
            source_layer_name, inner_index = source_layer
            tensor, inner_tensors = backbone_tensors[source_layer_name]

            if inner_index is None:
                input_tensor = backbone_tensors[source_layer_name][0]
            else:
                input_tensor = backbone_tensors[source_layer_name][1][inner_index]

            rh_layer = regression_headers[layer_index]
            ch_layer = classification_headers[layer_index]

            rh, ch = self.compute_headers(rh_layer(input_tensor) , ch_layer(input_tensor))
            regression_header_tensors.append(rh)
            classification_header_tensors.append(ch)

        locations = Concatenate(axis=1)(regression_header_tensors) 
        confidences = Concatenate(axis=1)(classification_header_tensors) 

        self.model = Model(inputs=input_layer, outputs=[confidences, locations])
        
    def compute_headers(self, rh_output, ch_output):
        res_shape = ch_output.shape[1]*ch_output.shape[2]*ch_output.shape[3]//self.n_class
        ch_output = Reshape((res_shape, self.n_class))(ch_output)

        res_shape = rh_output.shape[1]*rh_output.shape[2]*rh_output.shape[3]//4
        rh_output = Reshape((res_shape, 4))(rh_output)

        return rh_output, ch_output

    def summary(self):
        self.model.summary()