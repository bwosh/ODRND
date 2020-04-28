from models.core.architecture import NetArchitecture

class Net():
    def __init__(self, arch:NetArchitecture):
        self.arch = arch
        self.model = arch.to_model()

    def summary(self):
        return self.model.summary()