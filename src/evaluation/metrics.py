from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def build_classification_report(y_true, y_pred, target_names=None) -> str:
    return classification_report(y_true, y_pred, target_names=target_names)


def build_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
