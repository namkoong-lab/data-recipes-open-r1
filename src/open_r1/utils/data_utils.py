import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset


def uniform_simplex_sample(n_points, n_categories, from_dirichlet=True):
    """Sample a point from the uniform simplex in n_categories dimensions.
    Following: https://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex
    """
    if from_dirichlet:
        ps = np.random.dirichlet(np.ones(n_categories), size=n_points)
        return ps
    else:
        us = np.random.uniform(size=(n_points, n_categories))
        es = -np.log(us)
        return es / es.sum(axis=1, keepdims=True)


def analyze_types(types, verbose=False):
    type_counts = {}
    for type_ in types:
        if type_ not in type_counts:
            type_counts[type_] = 0
        type_counts[type_] += 1

    type_percentages = {}
    for type_, count in type_counts.items():
        type_percentages[type_] = count / len(types)

    if verbose:
        for type_, count in type_counts.items():
            print(f"{type_}:\n\t{count}\n\t{count / len(types) * 100:.2f}%")

    return type_percentages


def get_light_evals(proportions, num_train=2500, num_test=1000, seed=42):
    """
    Get the DigitalLearningGmbH/MATH-lighteval dataset
    """
    light_evals = load_dataset(
        "DigitalLearningGmbH/MATH-lighteval",
        "default",
    )
    ALL_LIGHT_EVAL_TYPES = [
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Precalculus",
        "Number Theory",
        "Prealgebra",
    ]
    TRAIN_EVAL_TYPES = ALL_LIGHT_EVAL_TYPES[
        :5
    ]  # Only use the first 5 types for training

    # subsample training dataset
    assert np.isclose(
        1.0, sum(proportions.values()), atol=1e-2
    ), "Proportions must sum to 1.0"

    # Sample the training dataset (w/ replacement)
    train_datasets = []
    for light_eval_type in TRAIN_EVAL_TYPES:
        n_samples = int(proportions[light_eval_type] * num_train)
        type_dataset = light_evals["train"].filter(
            lambda x: x["type"] == light_eval_type
        )

        assert len(type_dataset) > 0, f"Dataset for {light_eval_type} is empty"

        rng = np.random.default_rng(seed)
        random_indices = rng.integers(low=0, high=len(type_dataset), size=n_samples)

        sampled_dataset = type_dataset.select(random_indices)
        train_datasets.append(sampled_dataset)

    # Concatenate the sampled datasets
    train_dataset = concatenate_datasets(train_datasets)

    # Subsample test dataset (w/o replacement, uniform downsampling)
    test_datasets = {}
    test_downsampling_ratio = num_test / len(light_evals["test"])
    assert (
        test_downsampling_ratio <= 1.0
    ), f"Test downsampling ratio {test_downsampling_ratio} is greater than 1.0"
    for eval_type in ALL_LIGHT_EVAL_TYPES:
        type_dataset = light_evals["test"].filter(lambda x: x["type"] == eval_type)
        type_dataset = type_dataset.shuffle(seed=seed)
        sampled_dataset = type_dataset.select(
            range(int(len(type_dataset) * test_downsampling_ratio))
        )
        test_datasets[eval_type] = sampled_dataset

    # Concatenate the sampled datasets
    test_datasets = concatenate_datasets(
        [test_datasets[eval_type] for eval_type in ALL_LIGHT_EVAL_TYPES]
    )
    data_dict["test"] = test_datasets
    data_dict["train"] = train_dataset
    return DatasetDict(data_dict)
