import os
from tensorflow import keras

from models.core.architecture import NetArchitecture

class Net():
    def __init__(self, arch:NetArchitecture):
        self.arch = arch
        self.model = arch.to_model()

    def summary(self):
        return self.model.summary()

    def save(self, path):
        print(f"Saving model ({path}).")
        self.model.save_weights(path)

    def load(self, path):
        if os.path.isfile(path):
            print(f"Loading model ({path}).")
            self.model.load_weights(path)
