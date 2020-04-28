from training.sample import TrainingSample

class Dataset():
    def __init__(self, name):
        self.name = name

    def __len__(self):
        raise Exception("Not implmented")

    def __getitem__(self, index) -> TrainingSample:
        raise Exception("Not implmented")