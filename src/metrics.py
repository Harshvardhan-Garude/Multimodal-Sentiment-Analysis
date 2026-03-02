import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, macro-F1, per-class F1, and confusion matrix.

    Args:
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm
    }
