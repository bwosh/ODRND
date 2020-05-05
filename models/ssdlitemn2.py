from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from models.backbones.mobilenet import MNv2

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

        self.input = Input(shape=(self.input_size, self.input_size, 3))
        self.features = self.backbone.add_feature_layers(self.input)
        
        # TODO add ssd parts
        self.model = Model(inputs = self.input, outputs=self.features)