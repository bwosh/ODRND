import os

class Net():
    def __init__(self, model):
        self.model = model

    def summary(self):
        return self.model.model.summary()

    def save(self, path):
        print(f"Saving model ({path}).")
        self.model.model.save_weights(path)

    def save_full(self, path):
        print(f"Saving model ({path}).")
        self.model.model.save(path)

    def load(self, path):
        if os.path.isfile(path):
            print(f"Loading model ({path}).")
            self.model.model.load_weights(path)

    def detect(self, rgb_array):
        pass
        #TODO return self.model.detect(rgb_array)
