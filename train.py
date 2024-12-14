import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from pprint import pprint
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
import multiprocessing

import faiss
from models import cls_model
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (cls_dataset, FocalLoss, debug_shapes, 
                  read_dataset, preprocess_dataset, prepare_data,
                  download_and_extract_dataset, encode_title, generate_negative_samples)
from eval_model_utils import evaluation
def train_model(df_train, df_test, userCount, itemCount, device):
    """
    Train the recommendation model
    """
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/recommendation_experiment')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")


    epochs = 5
    all_unique_titles = pd.concat([df_train, df_test])["title"].unique()

    # Title to index mapping
    title_to_idx = {title: idx for idx, title in enumerate(all_unique_titles)}
    dict_titlEmb = encode_title(df_train,df_test)
    list_titlEmb = [dict_titlEmb[title] for title in title_to_idx]
    title_emb_tensor = torch.stack(list_titlEmb)

    # Dataset and DataLoader
    ds_train = cls_dataset(df_train, userCount, itemCount, title_emb_tensor, title_to_idx)
    ds_test = cls_dataset(df_test, userCount, itemCount, title_emb_tensor, title_to_idx)

    num_workers = multiprocessing.cpu_count()
    ds_trainLoader = DataLoader(ds_train, batch_size=512, num_workers=num_workers, pin_memory=True)
    ds_testLoader = DataLoader(ds_test, batch_size=1000, num_workers=num_workers, pin_memory=True)

    # Model initialization
    modelRec = cls_model(userCount, itemCount, user_embSize=32, item_embSize=32)
    if torch.cuda.device_count() > 1:
        modelRec = nn.DataParallel(modelRec)
    modelRec = modelRec.to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean').to(device)
    optim = torch.optim.Adam(modelRec.parameters(), lr=1e-3)

    # GradScaler for mixed precision
    scaler = torch.GradScaler()

    for epoch in range(epochs):
        modelRec.train()
        loss_acc = 0
        for i, (x1, x2, y, _) in enumerate(tqdm(ds_trainLoader, desc=f"Epoch {epoch+1}")):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad()
            with torch.autocast(device_type='cuda'):
                logits = modelRec(x1, x2)
                loss = criterion(logits, y.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            loss_acc += loss.item()

            # Optional: Log every N batches
            if (i + 1) % 100 == 0:
                avg_loss = loss_acc / 100
                print(f"Epoch {epoch+1}, Batch {i+1}, Train Loss: {avg_loss:.4f}")
                writer.add_scalar('Train/Loss', avg_loss, epoch * len(ds_trainLoader) + i + 1)
                loss_acc = 0

        # Validation at the end of each epoch
        modelRec.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x1_val, x2_val, y_val, _ in tqdm(ds_testLoader, desc="Validation"):
                x1_val = x1_val.to(device, non_blocking=True)
                x2_val = x2_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                with torch.autocast(device_type='cuda'):
                    logits_val = modelRec(x1_val, x2_val)
                    loss_val = criterion(logits_val, y_val.squeeze())

                val_loss += loss_val.item()
                preds = torch.argmax(logits_val, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_val.squeeze().cpu().numpy())

        avg_val_loss = val_loss / len(ds_testLoader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)

    writer.close()
    return modelRec
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Configuration
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Download dataset if not exists
    print("Checking and downloading dataset if needed...")
    download_and_extract_dataset()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    
    # Load and preprocess data
    df_movies, df_ratings, df_users = read_dataset()
    df_combined = preprocess_dataset(df_movies, df_ratings, df_users)
    
 
    # Determine userCount and itemCount
    max_user_id = df_combined["user_id"].max()
    max_item_id = df_combined["item_id"].max()
    userCount = max_user_id + 1
    itemCount = max_item_id + 1

    NEGATIVE_SAMPLES_PER_USER = 80
    # Prepare train and test datasets
    df_train, df_test = prepare_data(df_combined, df_movies, df_users)

    # Train model
    model = train_model(df_train, df_test, userCount, itemCount, device)

    # Save model
    torch.save(model.state_dict(), 'modelRec.pth')

    # Evaluation of the performances
    evaluation(userCount, itemCount, device, df_train,df_test, df_combined)

