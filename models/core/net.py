from models.core.builder import *

class Net():
    def __init__(self, model):
        self.model = model

    @staticmethod
    def from_arch(arch):
        # build keras model
        nodes = get_nodes(arch)
        keras_model = None

        # return wrapped model
        net = Net(keras_model)
        return net

    def summary(self):
        return "model-summary-to-be-done" # TODO
        #return self.model.summary()