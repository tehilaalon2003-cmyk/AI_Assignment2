from typing import List


# Calculates the classification accuracy between true and predicted labels
def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred")

    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1

    return correct / len(y_true) if y_true else 0.0
