# TODO training_cd generator - to be reviewed

import numpy as np

from tensorflow.keras.utils import Sequence

from datasets.base import Dataset
from models.core.net import NetArchitecture

class Generator(Sequence):
    def __init__(self, dataset:Dataset, architecture:NetArchitecture, opts):
        self.dataset = dataset

        self.batch_size = opts.batch_size
        self.opts = opts

        self.input_size = architecture.input_shapes["input"]
        self.target_size = architecture.output_sizes

        class_ids = list(set(dataset.category_mapping.values()))
        class_ids.sort()

        self.class_ids = class_ids
        self.index_map = np.arange(0,len(self.dataset), 1)

    def get_sample_data(self, item):
        img = item.get_image_as_rgb_array(self.input_size)/255
        hm = item.get_centers_heatmap(self.class_ids, self.target_size)
        wh, mask = item.get_height_width_maps(self.target_size)

        return [img, mask], [hm, wh]

    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, batch_index):
        all_X = []
        all_y = []

        for index in range(batch_index*self.batch_size, (batch_index+1)*self.batch_size,1):
            if index < len(self.dataset):
                item = self.dataset[self.index_map[index]]
                X, y = self.get_sample_data(item)
                all_X.append(X)
                all_y.append(y)

        X0_input = np.stack([x[0] for x in all_X])
        X1_input = np.stack([x[1] for x in all_X])
        y0_input = np.stack([y[0] for y in all_y])
        y1_input = np.stack([y[1] for y in all_y])

        return [X0_input, X1_input], [y0_input, y1_input]

    def on_epoch_end(self):
        np.random.shuffle(self.index_map)
