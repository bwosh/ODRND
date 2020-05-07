from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from models.backbones.mobilenet import MNv2
from models.detectors.ssd import SSD
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
        
        # TODO add ssd parts
        self.backbone = Model(inputs = self.input, outputs=self.features)
        source_layer_indexes = []
        extras, classification_headers, regression_headers = self.create_ssd_parts()
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
        # TODO ssd parts
        return None, None, None