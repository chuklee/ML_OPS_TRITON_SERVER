"""
Utility functions for the project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_results(results: pd.DataFrame, results_path: str, cross_validation: bool = False):
    """
    Plots the training and evaluation results.

    Args:
        results: The training and evaluation results.
        results_path: The path to save the plots.
        cross_validation: Whether the results are from cross-validation.
    """
    if cross_validation:
        # Plot metrics for each fold
        for fold in results['fold'].unique():
            fold_results = results[results['fold'] == fold]
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            sns.lineplot(x='epoch', y='loss_train', data=fold_results, label='loss_train', ax=ax[0])
            sns.lineplot(x='epoch', y='loss_test', data=fold_results, label='loss_test', ax=ax[0])
            sns.lineplot(x='epoch', y='auc_train', data=fold_results, label='auc_train', ax=ax[1])
            sns.lineplot(x='epoch', y='auc_test', data=fold_results, label='auc_test', ax=ax[1])
            ax[0].set_title(f'Fold {fold} - Loss')
            ax[1].set_title(f'Fold {fold} - AUC')
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, f'cv_results_fold_{fold}.png'))
            plt.close()

        # Plot average metrics across folds
        avg_results = results.groupby('epoch').mean().reset_index()
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.lineplot(x='epoch', y='loss_train', data=avg_results, label='loss_train', ax=ax[0])
        sns.lineplot(x='epoch', y='loss_test', data=avg_results, label='loss_test', ax=ax[0])
        sns.lineplot(x='epoch', y='auc_train', data=avg_results, label='auc_train', ax=ax[1])
        sns.lineplot(x='epoch', y='auc_test', data=avg_results, label='auc_test', ax=ax[1])
        ax[0].set_title('Average Loss Across Folds')
        ax[1].set_title('Average AUC Across Folds')
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'cv_results_average.png'))
        plt.close()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.lineplot(x='epoch', y='loss_train', data=results, label='loss_train', ax=ax[0])
        sns.lineplot(x='epoch', y='loss_test', data=results, label='loss_test', ax=ax[0])
        sns.lineplot(x='epoch', y='auc_train', data=results, label='auc_train', ax=ax[1])
        sns.lineplot(x='epoch', y='auc_test', data=results, label='auc_test', ax=ax[1])
        ax[0].set_title('Loss')
        ax[1].set_title('AUC')
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'results.png'))
        plt.close()

def plot_feature_engineering(df: pd.DataFrame, results_path: str):
    """
    Plots various aspects of feature engineering for a given DataFrame.

    Args:
        df: The input DataFrame.
        results_path: The path to save the generated plots.
    """
    # Select only numerical columns for correlation calculation
    numerical_df = df.select_dtypes(include=np.number)

    # Compute the correlation matrix for numerical columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(results_path, "correlation_matrix.png"))
    plt.close()

    # Example: Distribution of a numerical feature
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], kde=True)
    plt.title('Distribution of Age')
    plt.savefig(os.path.join(results_path, 'age_distribution.png'))
    plt.show()

    # Example: Count plot of a categorical feature
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Count of Gender')
    plt.savefig(os.path.join(results_path, 'gender_count.png'))
    plt.show()