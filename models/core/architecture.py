class NetArchitecture:
    def __init__(self, arch_definition, outputs):
        self.arch_definition = arch_definition
        self.outputs = outputs

    def __len__(self):
        return len(self.arch_definition)

    def __getitem__(self, index):
        return self.arch_definition[index]