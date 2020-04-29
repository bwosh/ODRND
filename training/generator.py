import numpy as np

from datasets.base import Dataset

class Generator():
    def __init__(self, dataset:Dataset):
        self.dataset = dataset

        self.batch_size = 4

        self.input_size = (224,224,3) # TODO move input size elsewhere
        self.target_size = (32,32, 3) # TODO move input size elsewhere
        self.class_ids = [0,1,2]

    def get_sample_data(self, item):
        img = item.get_image_as_rgb_array(self.input_size)
        hm = item.get_centers_heatmap(self.class_ids, self.target_size)
        wh, mask = item.get_height_width_maps(self.target_size)

        return [img, mask], [hm, wh]

    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        X, y = get_sample_data(index)
        return X, y

    def on_epoch_end(self):
        # TODO shuffle data
        pass

    def __data_generation(self, list_IDs_temp):
        # TODO __data_generation
        pass