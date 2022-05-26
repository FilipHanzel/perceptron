import random
from typing import List, Tuple, Union, Dict


def transpose(data: List[List]) -> List[List]:
    return [list(column) for column in zip(*data)]


def shuffle(features: List, targets: List) -> Tuple[List, List]:
    order = list(range(len(features)))
    random.shuffle(order)

    features = [features[index] for index in order]
    targets = [targets[index] for index in order]

    return features, targets


def kfold_split(
    inputs: List[List[float]],
    targets: List[List[float]],
    fold_count: int,
    stratified: bool = True,
    random: bool = True,
) -> Dict:

    if random:
        inputs, targets = shuffle(inputs, targets)

    if stratified:
        records = sorted(zip(targets, inputs), key=lambda x: x[0])
    else:
        records = zip(targets, inputs)

    folds = [dict(inputs=[], targets=[]) for _ in range(fold_count)]

    for index, (target, inp) in enumerate(records):
        fold_idx = index % fold_count
        folds[fold_idx]["inputs"].append(inp)
        folds[fold_idx]["targets"].append(target)

    return folds


def to_binary(column: List[str]) -> List[List[int]]:
    """Encode column values as 0 or 1."""
    values = sorted(set(column))
    assert len(values) == 2, "Too many values for binary encoding"

    mapping = {label: index for index, label in enumerate(values)}
    encoded = [[mapping[value]] for value in column]

    return mapping, encoded


def to_categorical(column: List[str]) -> List[List[int]]:
    """Encode column values as binary vectors."""
    values = sorted(set(column))

    index_mapping = {label: index for index, label in enumerate(values)}

    categorical = [[0] * len(values) for _ in range(len(column))]
    for vector, value in zip(categorical, column):
        vector[index_mapping[value]] = 1

    mapping = {label: [0] * len(values) for label in index_mapping}
    for label, index in index_mapping.items():
        mapping[label][index] = 1

    return mapping, categorical
