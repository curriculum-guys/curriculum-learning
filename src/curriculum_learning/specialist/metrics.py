from sklearn.metrics import confusion_matrix

class SpecialistMetrics:
    def __init__(self, labels, matrix_size) -> None:
        self.labels = labels
        self.matrix_size = matrix_size

    def format_matrix(self, name, **values):
        matrix = {}
        expected = ['tn', 'fp', 'fn', 'tp']
        for key in expected:
            matrix[name + '_' + key] = values.get(key)
        return matrix

    def build_matrix(self, name, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=self.labels).ravel()
        return self.format_matrix(name, tn=tn, fp=fp, fn=fn, tp=tp)

    def null_matrix(self, name):
        return self.format_matrix(
            name, tn=None, fp=None, fn=None, tp=None
        )
