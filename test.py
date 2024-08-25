import models
import datasets

import random
import math
import torch
from torch import nn
import numpy as np

N_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
N_DIM = 128
SOURCE_LENGTH = 200
TARGET_LENGTH = 4

device = 'cuda'

def top_k_acc(preds, target, k):
    batch_size = preds.shape[0]
    n_tickers = preds.shape[2]
    target_diffs = target[:, -1, :] - target[:, 0, :]
    preds_diffs = preds[:, -1, :] - preds[:, 0, :]
    predicted_indices = np.argmax(preds_diffs, axis=1)
    actual_kth_high_diffs = np.partition(target_diffs, -k, axis=1)[:, -k]
    correct = 0
    for i in range(batch_size):
        """
        print(f'kth actual: {actual_kth_high_diffs[i]}')
        print(f'picked: {target_diffs[i, predicted_indices[i]]}')
        """
        if actual_kth_high_diffs[i] <= target_diffs[i, predicted_indices[i]]:
            correct += 1
    return correct / batch_size

def main():
    torch.set_printoptions(sci_mode=False)
    
    train_data = datasets.DayCloseDataset('2010-01-04', '2022-08-21', N_DIM, 5, SOURCE_LENGTH, TARGET_LENGTH)
    val_data = datasets.DayCloseDataset('2022-08-22', '2024-08-21', N_DIM, 5, SOURCE_LENGTH, TARGET_LENGTH)
    
    
    train_loader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, BATCH_SIZE, shuffle=True)
    
    model = models.PredTransformer(N_DIM).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_fn = nn.MSELoss()
    
    for _ in range(N_EPOCHS):
        total_loss = 0
        for input, target in train_loader:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            for i in range(TARGET_LENGTH):
                if i == 0:
                    pred = model(input)[:, None, :]
                    preds = pred
                else:
                    pred = model(torch.cat([input, preds], dim=1))[:, None, :]
                    preds = torch.cat([preds, pred], dim=1)
            """
            if random.random() < 0.01:
                print(target[0, :, 0])
                print(preds[0, :, 0])
                """
            loss = loss_fn(preds, target)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print(f'train loss: {total_loss / len(train_loader)}')
        
        total_loss = 0
        with torch.no_grad():
            total_top_10_acc = 0
            for input, target in val_loader:
                input = input.to(device)
                target = target.to(device)
                for i in range(TARGET_LENGTH):
                    if i == 0:
                        pred = model(input)[:, None, :]
                        preds = pred
                    else:
                        pred = model(torch.cat([input, preds], dim=1))[:, None, :]
                        preds = torch.cat([preds, pred], dim=1)
                """
                if random.random() < 0.01:
                    print(torch.cat([input[0, :, 0], target[0, :, 0]]))
                    print(preds[0, :, 0])
                    """
                loss = loss_fn(preds, target)
                total_loss += loss.item()
                total_top_10_acc += top_k_acc(preds.cpu().numpy(), target.cpu().numpy(), 10)

        print(f'val loss: {total_loss / len(val_loader)}')
        print(f'top-10 acc delta: {total_top_10_acc / len(val_loader) - 10 / N_DIM}')

if __name__ == '__main__':
    main()