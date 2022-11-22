from sklearn.neural_network import MLPClassifier

class SpecialistModel:
    def __init__(self, labels=['bad', 'good']):
        self.instance = MLPClassifier(**self.params, random_state=42)
        self.type = 'mlp_classifier'
        self.labels = labels
        self.fitted = False

    @property
    def params(self):
        return {
            'activation': 'relu',
            'alpha': 0.001,
            'hidden_layer_sizes': (256,128,256),
            'solver': 'adam'
        }

    def predict(self, X):
        return self.instance.predict(X)

    def score(self, X, y):
        return self.instance.score(X, y)

    def fit(self, X, y):
        self.instance = self.instance.partial_fit(X, y, self.labels)
        self.fitted = True
