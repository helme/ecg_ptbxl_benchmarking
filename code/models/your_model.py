from models.base_model import ClassificationModel

class YourModel(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape 

    def fit(self, X_train, y_train, X_val, y_val):
        pass

    def predict(self, X):
        pass