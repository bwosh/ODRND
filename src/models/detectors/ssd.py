class SSD:
    def __init__(self, n_class, backbone, source_layer_indexes,
                         extras, classification_headers, regression_headers):
        # TODO implement SSD here
        self.model = backbone

    def summary(self):
        self.model.summary() # TODO check if ok later