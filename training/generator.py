import numpy as np

from tensorflow.keras.utils import Sequence

from datasets.base import Dataset
from models.core.net import NetArchitecture

class Generator(Sequence):
    def __init__(self, dataset:Dataset, architecture:NetArchitecture):
        self.dataset = dataset

        self.batch_size = 4

        self.input_size = architecture.input_shapes["input"]
        self.target_size = architecture.output_sizes
        self.class_ids = [0,1,2]

    def get_sample_data(self, item):
        img = item.get_image_as_rgb_array(self.input_size)
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
                item = self.dataset[index]
                X, y = self.get_sample_data(item)
                all_X.append(X)
                all_y.append(y)

        X0_input = np.stack([x[0] for x in all_X])
        X1_input = np.stack([x[1] for x in all_X])
        y0_input = np.stack([y[0] for y in all_y])
        y1_input = np.stack([y[1] for y in all_y])

        #print(X0_input.shape)
        #print(X1_input.shape)
        #print(y0_input.shape)
        #print(y1_input.shape)
        #exit(0)

        return [X0_input, X1_input], [y0_input, y1_input]

    def on_epoch_end(self):
        print("on_epoch_end")
        # TODO shuffle data
        pass

    def __data_generation(self, list_IDs_temp):
        print("__data_generation")
        # TODO __data_generation
        pass