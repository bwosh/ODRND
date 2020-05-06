from datasets.sample import DatasetSample

class Dataset():
    def __init__(self, name):
        self.name = name

    def __len__(self):
        raise Exception("Not implmented")

    def __getitem__(self, index) -> DatasetSample:
        raise Exception("Not implmented")