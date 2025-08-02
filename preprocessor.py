from preprocessor_methods import PreprocessorMethod

class Preprocessor:
    def __init__(self, methods: list[PreprocessorMethod]):
        self.methods = methods
    
    def fit_transform(self, train_inputs, train_teachers):
        for method in self.methods:
            method.fit(train_inputs, train_teachers)
            train_inputs, train_teachers = method.transform(train_inputs, train_teachers)
        return train_inputs, train_teachers
        
    def transform(self, inputs, teachers):
        for method in self.methods:
            inputs, teachers = method.transform(inputs, teachers)
        return inputs, teachers