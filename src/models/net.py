import numpy as np
import os

from models.net import Net

class Net():
    def __init__(self, model: Net):
        self.model = model

    def summary(self):
        return self.model.model.summary()

    def save(self, path):str:
        print(f"Saving model ({path}).")
        self.model.model.save_weights(path)

    def save_full(self, path:str)::
        print(f"Saving model ({path}).")
        self.model.model.save(path)

    def load(self, path:str):
        if os.path.isfile(path):
            print(f"Loading model ({path}).")
            self.model.model.load_weights(path)

    def detect(self, rgb_array:np.ndarray):
        return self.model.detect(rgb_array)
