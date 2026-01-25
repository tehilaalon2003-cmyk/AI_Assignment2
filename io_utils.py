from dataclasses import dataclass
from typing import List, Dict


# Data structure that holds a dataset after reading it from a file
@dataclass
class Dataset:
    header: List[str]
    X: List[Dict[str, str]]
    y: List[str]


# Reads a TAB-separated dataset file and returns it in a structured format
def read_tsv_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split("\t")
    feature_names = header[:-1]

    X = []
    y = []

    for line in lines[1:]:
        values = line.split("\t")

        if len(values) != len(header):
            raise ValueError("Row length does not match header")

        sample = {}
        for i, fname in enumerate(feature_names):
            sample[fname] = values[i]

        X.append(sample)
        y.append(values[-1])

    return Dataset(header=header, X=X, y=y)


# Writes predictions and accuracy results to the output file
def write_output(
    path: str,
    dt_preds: List[str],
    nb_preds: List[str],
    dt_acc: float,
    nb_acc: float
):
    if len(dt_preds) != len(nb_preds):
        raise ValueError("Prediction lists must have the same length")

    with open(path, "w", encoding="utf-8") as f:
        for dt, nb in zip(dt_preds, nb_preds):
            f.write(f"{dt}\t{nb}\n")

        f.write(f"{dt_acc}\t{nb_acc}\n")
