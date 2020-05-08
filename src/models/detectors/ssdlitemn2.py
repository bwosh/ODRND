from tensorflow.keras.layers import Input, SeparableConv2D, Conv2D
from tensorflow.keras.models import Model

from models.backbones.mobilenet import MNv2
from models.detectors.ssd import SSD, GraphPath
from models.utils.box_utils import SSDBoxSizes, SSDSpec, generate_ssd_priors

class SSDLiteMN2:
    def __init__(self, n_class=3, input_size=300, width_mult=1.,use_batch_norm=True, input_channel=32, last_channel=1280):
        self.backbone = MNv2(
            n_class=n_class, 
            input_size=input_size, 
            width_mult=width_mult,
            use_batch_norm=use_batch_norm, 
            input_channel=input_channel, 
            last_channel=last_channel, 
            build_model=False)

        self.input_size = input_size
        self.init_ssdspec()

        self.input = Input(shape=(self.input_size, self.input_size, 3))
        self.features = self.backbone.add_feature_layers(self.input)
        self.n_class = n_class
        
        self.backbone = Model(inputs = self.input, outputs=self.features)
        source_layer_indexes, extras, classification_headers, regression_headers = self.create_ssd_parts()
        self.model = SSD(n_class, self.backbone, source_layer_indexes,
                         extras, classification_headers, regression_headers)

    def init_ssdspec(self):
        
        self.iou_threshold = 0.45
        self.center_variance = 0.1
        self.size_variance = 0.2

        self.ssdspecs = [
            SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
            SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
            SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
            SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
            SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
            SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
        ]

        # N*(x,y,w,h)
        self.priors = generate_ssd_priors(self.ssdspecs, self.input_size)
  
    def create_ssd_parts(self):

        source_layer_indexes = [
            GraphPath(14, 'conv', 3),
            19,
        ]

        extras = [
            lambda x: inverted_residual("E0", x, 1280, 512, stride=2, expand_ratio=0.2),
            lambda x: inverted_residual("E1", x, 512, 256, stride=2, expand_ratio=0.25),
            lambda x: inverted_residual("E2", x, 256, 256, stride=2, expand_ratio=0.5),
            lambda x: inverted_residual("E3", x, 256, 54, stride=2, expand_ratio=0.25),

        ]

        regression_headers = [
            SeparableConv2D(6*4, kernel_size=3, padding='same', name="RH0"),
            SeparableConv2D(6*4, kernel_size=3, padding='same', name="RH1"),
            SeparableConv2D(6*4, kernel_size=3, padding='same', name="RH2"),
            SeparableConv2D(6*4, kernel_size=3, padding='same', name="RH3"),
            SeparableConv2D(6*4, kernel_size=3, padding='same', name="RH4"),
            Conv2D(6*4, kernel_size=1, padding='valid', name="RH5"),
        ]

        classification_headers = [
            SeparableConv2D(6*self.n_class, kernel_size=3, padding='same', name="CH0"),
            SeparableConv2D(6*self.n_class, kernel_size=3, padding='same', name="CH1"),
            SeparableConv2D(6*self.n_class, kernel_size=3, padding='same', name="CH2"),
            SeparableConv2D(6*self.n_class, kernel_size=3, padding='same', name="CH3"),
            SeparableConv2D(6*self.n_class, kernel_size=3, padding='same', name="CH4"),
            Conv2D(6*self.n_class, kernel_size=1, padding='valid', name="CH5"),
        ]

        return source_layer_indexes, extras, classification_headers, regression_headers