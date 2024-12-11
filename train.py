"""
Main script for training and evaluating a DLRM model on the MovieLens dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from batch import RecBatch
from model import DLRMCustom
from utils import plot_results, plot_feature_engineering
import plotly.graph_objects as go

# --- Constants ---
DATA_PATH = "data/ml-1m"
RESULTS_PATH = "results/figures"
COLS_DENSE = ["Age"]
COLS_SPARSE = ["UserID", "MovieID", "Gender", "Occupation", "Zip-code", "Genres"]
EMBEDDING_DIM = 256
DENSE_ARCH_LAYER_SIZES = [512, 256, EMBEDDING_DIM]
OVER_ARCH_LAYER_SIZES = [512, 512, 256, 1]
ADAGRAD = False
EPS = 1e-8
LEARNING_RATE = 0.01
N_EPOCHS = 100
E_PATIENCE = 100
BATCH_SIZE = 500
NUM_GENERATED_BATCHES_TRAIN = None
NUM_GENERATED_BATCHES_VAL = None
SEED = 123
N_SPLITS = 3
DATA_PERCENTAGE = 1

# --- Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{DEVICE}'")

# --- Functions ---
def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from the movielens dataset.

    Args:
        data_path: The path to the movielens dataset.
    Returns:
        movies_data: The movies data.
        users_data: The users data.
        ratings_data: The ratings data.
    """
    movies_data: pd.DataFrame = pd.read_csv(
        os.path.join(data_path, "movies.dat"),
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding='latin1'
    )
    users_data: pd.DataFrame = pd.read_csv(
        os.path.join(data_path, "users.dat"),
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding='latin1'
    )
    ratings_data: pd.DataFrame = pd.read_csv(
        os.path.join(data_path, "ratings.dat"),
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding='latin1'
    )
    return movies_data, users_data, ratings_data

def preprocess_data(movies_df: pd.DataFrame, users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data by merging, creating labels, and splitting into train, validation, and test sets.

    Args:
        movies_df: The movies DataFrame.
        users_df: The users DataFrame.
        ratings_df: The ratings DataFrame.

    Returns:
        train_df: The preprocessed training DataFrame.
        val_df: The preprocessed validation DataFrame.
        test_df: The preprocessed test DataFrame.
    """
    train_df = pd.merge(ratings_df, users_df, on="UserID")
    train_df = pd.merge(train_df, movies_df, on="MovieID")
    train_df["label"] = (train_df["Rating"] >= 4).astype(int)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["label"])
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=SEED, stratify=test_df["label"])

    # Reduce data based on DATA_PERCENTAGE
    train_df = train_df.sample(frac=DATA_PERCENTAGE, random_state=SEED)
    val_df = val_df.sample(frac=DATA_PERCENTAGE, random_state=SEED)
    test_df = test_df.sample(frac=DATA_PERCENTAGE, random_state=SEED)

    return train_df, val_df, test_df

def encode_sparse_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cols_sparse: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Encodes sparse features using a mapping dictionary.

    Args:
        train_df: The training DataFrame.
        val_df: The validation DataFrame.
        test_df: The test DataFrame.
        cols_sparse: The list of sparse columns.

    Returns:
        train_df: The training DataFrame with encoded sparse features.
        val_df: The validation DataFrame with encoded sparse features.
        test_df: The test DataFrame with encoded sparse features.
        map_sparse: The mapping dictionary for sparse features.
        map_sparse_rev: The reverse mapping dictionary for sparse features.
    """
    def _encode(item, map_rev, unk_index):
        if isinstance(item, (list, np.ndarray)):
            return [map_rev.get(x, unk_index) for x in item]
        elif isinstance(item, str) and "|" in item:
            parts = item.split("|")
            return [map_rev.get(part, unk_index) for part in parts]
        else:
            return [map_rev.get(item, unk_index)]

    map_sparse = {}
    map_sparse_rev = {}

    for feat in cols_sparse:
        if feat == "Genres":
            unique_genres = set()
            for genres in train_df[feat]:
                unique_genres.update(genres.split("|"))
            map_sparse[feat] = {i: c for i, c in enumerate(sorted(unique_genres))}
        else:
            unique_values = train_df[feat].unique()
            map_sparse[feat] = {i: c for i, c in enumerate(sorted(unique_values))}

        map_sparse_rev[feat] = {v: k for k, v in map_sparse[feat].items()}
        unk_index = len(map_sparse[feat])
        map_sparse_rev[feat]['<UNK>'] = unk_index

        train_df[feat + '_enc'] = train_df[feat].apply(lambda x: _encode(x, map_sparse_rev[feat], unk_index))
        test_df[feat + '_enc'] = test_df[feat].apply(lambda x: _encode(x, map_sparse_rev[feat], unk_index))
        val_df[feat + '_enc'] = val_df[feat].apply(lambda x: _encode(x, map_sparse_rev[feat], unk_index))

    return train_df, val_df, test_df, map_sparse, map_sparse_rev

def create_data_batches(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cols_sparse: list[str], cols_dense: list[str], batch_size: int, num_generated_batches: int, seed: int, device: torch.device) -> tuple[RecBatch, RecBatch, RecBatch]:
    """
    Creates data batches for training, validation, and testing.

    Args:
        train_df: The training DataFrame.
        val_df: The validation DataFrame.
        test_df: The test DataFrame.
        cols_sparse: The list of sparse columns.
        cols_dense: The list of dense columns.
        batch_size: The batch size.
        num_generated_batches: The number of generated batches.
        seed: The random seed.
        device: The device to use.

    Returns:
        train_data: The training data batch.
        val_data: The validation data batch.
        test_data: The test data batch.
    """
    train_data = RecBatch(
        data=train_df,
        cols_sparse=[c + '_enc' for c in cols_sparse],
        cols_dense=cols_dense,
        col_label="label",
        batch_size=batch_size,
        num_generated_batches=num_generated_batches,
        seed=seed,
        device=device
    )
    val_data = RecBatch(
        data=val_df,
        cols_sparse=[c + '_enc' for c in cols_sparse],
        cols_dense=cols_dense,
        col_label="label",
        batch_size=batch_size,
        num_generated_batches=num_generated_batches,
        seed=seed,
        device=device
    )
    test_data = RecBatch(
        data=test_df,
        cols_sparse=[c + '_enc' for c in cols_sparse],
        cols_dense=cols_dense,
        col_label="label",
        batch_size=batch_size,
        num_generated_batches=num_generated_batches,
        seed=seed,
        device=device
    )
    return train_data, val_data, test_data

def validate_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cols_sparse: list[str], num_embeddings_per_feature: dict):
    """
    Validates the data by checking for NaNs and ensuring that the maximum encoded values are within the embedding limits.

    Args:
        train_df: The training DataFrame.
        val_df: The validation DataFrame.
        test_df: The test DataFrame.
        cols_sparse: The list of sparse columns.
        num_embeddings_per_feature: The number of embeddings per feature.
    """
    print("Training DataFrame contains NaN:", train_df.isnull().values.any())
    print("Validation DataFrame contains NaN:", val_df.isnull().values.any())
    print("Test DataFrame contains NaN:", test_df.isnull().values.any())

    for feat in cols_sparse:
        max_train = train_df[feat + '_enc'].apply(max).max()
        max_test = test_df[feat + '_enc'].apply(max).max()
        max_val = val_df[feat + '_enc'].apply(max).max()
        num_embeddings = num_embeddings_per_feature[feat + '_enc']
        print(f"Feature '{feat}': max_train={max_train}, max_test={max_test}, max_val={max_val}, num_embeddings={num_embeddings}")
        assert max_train < num_embeddings, f"Train max index {max_train} >= num_embeddings {num_embeddings} for feature {feat}"
        assert max_test < num_embeddings, f"Test max index {max_test} >= num_embeddings {num_embeddings} for feature {feat}"
        assert max_val < num_embeddings, f"Val max index {max_val} >= num_embeddings {num_embeddings} for feature {feat}"

def plot_feature_importance(model, map_sparse, results_path):
    """
    Plots the feature importances based on the magnitude of embedding weights.

    Args:
        model: The trained DLRM model.
        map_sparse: The mapping dictionary for sparse features.
        results_path: The path to save the feature importance plot.
    """
    # Extract embedding weights from the model
    embedding_weights = {}

    # Access embedding bag configs
    eb_configs = model.train_model.model.sparse_arch.embedding_bag_collection._embedding_bag_configs

    # Iterate through embedding bags and their corresponding configs
    for config, table in zip(eb_configs, model.train_model.model.sparse_arch.embedding_bag_collection.embedding_bags.values()):
        embedding_weights[config.name] = table.weight.data.cpu().numpy()

    # Calculate feature importances as the average magnitude of embedding vectors
    feature_importances = {}
    for feature_name, table_name in map_sparse.items():
        feature_importances[feature_name] = np.mean(
            np.abs(embedding_weights[f"t_{feature_name}"])
        )

    # Create a bar chart of feature importances
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(feature_importances.keys()),
                y=list(feature_importances.values()),
            )
        ]
    )
    fig.update_layout(title_text="Feature Importances", xaxis_title="Feature", yaxis_title="Importance")
    fig.write_image(os.path.join(results_path, "feature_importance.png"))

def train_and_evaluate(
    train_data: RecBatch,
    test_data: RecBatch,
    num_embeddings_per_feature: dict,
    map_sparse: dict,
    device: torch.device,
) -> pd.DataFrame:
    """
    Trains and evaluates the DLRM model.

    Args:
        train_data: The training data batch.
        test_data: The test data batch.
        num_embeddings_per_feature: The number of embeddings per feature.
        map_sparse: The mapping dictionary for sparse features.
        device: The device to use.

    Returns:
        results: The training and evaluation results.
    """
    model_dlrm = DLRMCustom(
        COLS_DENSE,
        COLS_SPARSE,
        EMBEDDING_DIM,
        num_embeddings_per_feature,
        DENSE_ARCH_LAYER_SIZES,
        OVER_ARCH_LAYER_SIZES,
        ADAGRAD,
        LEARNING_RATE,
        EPS,
        device,
    )

    scores = model_dlrm.train_test(train_data, test_data, N_EPOCHS, E_PATIENCE)
    results = pd.DataFrame(scores).T.reset_index().rename(columns={"index": "epoch"})

    # --- Plot feature importance ---
    plot_feature_importance(model_dlrm, map_sparse, RESULTS_PATH)

    plot_results(results, RESULTS_PATH)
    return results

def cross_validate(train_df: pd.DataFrame, num_embeddings_per_feature: dict, device: torch.device) -> pd.DataFrame:
    """
    Performs cross-validation on the DLRM model.

    Args:
        train_df: The training DataFrame.
        num_embeddings_per_feature: The number of embeddings per feature.
        device: The device to use.

    Returns:
        cv_results: The cross-validation results.
    """
    kfolds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_results = []

    for i, (train_index, test_index) in enumerate(kfolds.split(train_df, train_df.label)):
        print("---" * 10)
        print(f"Fold {i + 1}/{kfolds.n_splits}")

        train_df_kf = train_df.iloc[train_index].reset_index(drop=True)
        test_df_kf = train_df.iloc[test_index].reset_index(drop=True)

        # Reduce data based on DATA_PERCENTAGE
        train_df_kf = train_df_kf.sample(frac=DATA_PERCENTAGE, random_state=SEED)
        test_df_kf = test_df_kf.sample(frac=DATA_PERCENTAGE, random_state=SEED)

        print('   Generate train data ...')
        train_data = RecBatch(
            data=train_df_kf,
            cols_sparse=[c + '_enc' for c in COLS_SPARSE],
            cols_dense=COLS_DENSE,
            col_label="label",
            batch_size=BATCH_SIZE,
            num_generated_batches=NUM_GENERATED_BATCHES_TRAIN,
            seed=SEED,
            device=device
        )

        print('   Generate test data ...')
        test_data = RecBatch(
            data=test_df_kf,
            cols_sparse=[c + '_enc' for c in COLS_SPARSE],
            cols_dense=COLS_DENSE,
            col_label="label",
            batch_size=BATCH_SIZE,
            num_generated_batches=NUM_GENERATED_BATCHES_VAL,
            seed=SEED,
            device=device
        )

        model_dlrm = DLRMCustom(
            COLS_DENSE, COLS_SPARSE,
            EMBEDDING_DIM, num_embeddings_per_feature,
            DENSE_ARCH_LAYER_SIZES, OVER_ARCH_LAYER_SIZES,
            ADAGRAD, LEARNING_RATE, EPS,
            device
        )

        scores = model_dlrm.train_test(train_data, test_data, N_EPOCHS, E_PATIENCE)
        scores = pd.DataFrame(scores).T.reset_index().rename(columns={'index': 'epoch'})

        print('   Scores:')
        print(scores.iloc[-5:])

        # Store all metrics for each fold
        fold_results = scores.copy()
        fold_results['fold'] = i + 1
        cv_results.append(fold_results)

    # Concatenate results from all folds
    cv_results = pd.concat(cv_results, axis=0).reset_index(drop=True)
    return cv_results
    return test_aucs

def main():
    """
    Main function to run the DLRM model training and evaluation.
    """
    # --- Load and preprocess data ---
    movies_df, users_df, ratings_df = load_data(DATA_PATH)
    train_df, val_df, test_df = preprocess_data(movies_df, users_df, ratings_df)
    train_df, val_df, test_df, map_sparse, map_sparse_rev = encode_sparse_features(train_df, val_df, test_df, COLS_SPARSE)

    # --- Create data batches ---
    train_data, val_data, test_data = create_data_batches(
        train_df, val_df, test_df,
        COLS_SPARSE, COLS_DENSE,
        BATCH_SIZE, NUM_GENERATED_BATCHES_TRAIN,
        SEED, DEVICE
    )

    # --- Validate data ---
    num_embeddings_per_feature = {c + '_enc': len(v) + 1 for c, v in map_sparse.items()}
    validate_data(train_df, val_df, test_df, COLS_SPARSE, num_embeddings_per_feature)

    # --- Feature engineering plots ---
    os.makedirs(RESULTS_PATH, exist_ok=True)
    plot_feature_engineering(train_df, RESULTS_PATH)

    # --- Train and evaluate ---
    print("Training and evaluating the model...")
    results = train_and_evaluate(train_data, test_data, num_embeddings_per_feature, map_sparse, DEVICE)

    # --- Plot results ---
    plot_results(results, RESULTS_PATH)

    # --- Cross-validate ---
    print("Performing cross-validation...")
    train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    cv_results = cross_validate(train_df, num_embeddings_per_feature, DEVICE)

    # --- Plot cross-validation results ---
    plot_results(cv_results, RESULTS_PATH, cross_validation=True)

if __name__ == "__main__":
    main()