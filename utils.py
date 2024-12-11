"""
Utility functions for the project.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results: pd.DataFrame):
    """
    Plots the training and evaluation results.

    Args:
        results: The DataFrame containing the results.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(results['epoch'], results['loss_train'], label='loss_train')
    ax[0].plot(results['epoch'], results['losses_test'], label='losses_test')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(results['epoch'], results['auc_train'], label='auc_train')
    ax[1].plot(results['epoch'], results['auc_test'], label='auc_test')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('AUC')
    ax[1].legend()

    plt.show()