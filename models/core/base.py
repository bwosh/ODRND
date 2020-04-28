class BlockBase:
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

    def forward(self, x:list):
        raise Exception("Not implemented")
