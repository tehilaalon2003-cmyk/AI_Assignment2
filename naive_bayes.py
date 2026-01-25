import math
from typing import Dict, List


# Trains a Naive Bayes classifier with Laplace smoothing
def train_naive_bayes(
    X: List[Dict[str, str]],
    y: List[str],
    feature_names: List[str],
    alpha: float = 1.0
) -> Dict:
    
   # Train a categorical Naive Bayes classifier.
    

    classes = list(set(y))
    n_samples = len(y)

    class_counts = {c: 0 for c in classes}
    for label in y:
        class_counts[label] += 1

    priors = {
        c: math.log(class_counts[c] / n_samples)
        for c in classes
    }

    feature_domains = {f: set() for f in feature_names}
    for row in X:
        for f in feature_names:
            feature_domains[f].add(row[f])

    counts = {
        c: {
            f: {v: 0 for v in feature_domains[f]}
            for f in feature_names
        }
        for c in classes
    }

    for row, label in zip(X, y):
        for f in feature_names:
            value = row[f]
            counts[label][f][value] += 1

    likelihoods = {c: {} for c in classes}

    for c in classes:
        likelihoods[c] = {}
        for f in feature_names:
            likelihoods[c][f] = {}
            V = len(feature_domains[f])
            denom = class_counts[c] + alpha * V

            for v in feature_domains[f]:
                num = counts[c][f][v] + alpha
                likelihoods[c][f][v] = math.log(num / denom)

    return {
        "classes": classes,
        "priors": priors,
        "likelihoods": likelihoods,
        "features": feature_names
    }


# Predicts the class label for a single sample
def predict_naive_bayes(model: Dict, sample: Dict[str, str]) -> str:
    
   # Predict class label for one sample.


    best_class = None
    best_score = float("-inf")

    for c in model["classes"]:
        score = model["priors"][c]

        for f in model["features"]:
            value = sample[f]
            score += model["likelihoods"][c][f][value]

        if score > best_score:
            best_score = score
            best_class = c

    return best_class
