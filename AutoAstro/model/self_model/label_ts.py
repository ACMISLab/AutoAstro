#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/1 15:42
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error



def preprocess_data(X, Y):
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_scaled, Y_scaled = scaler_x.fit_transform(X), scaler_y.fit_transform(Y)
    return X_scaled, Y_scaled, scaler_x, scaler_y

def split_data(X, Y, labels, test_size=0.2):
    return train_test_split(X, Y, labels, test_size=test_size, random_state=42)

def prepare_tensors(X_train, X_test, Y_train, Y_test, device):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device).view(X_train.shape[0], 1, X_train.shape[1])
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device).view(X_test.shape[0], 1, X_test.shape[1])
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)
    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor

def get_data_loaders(X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, batch_size=4096):
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=300, patience=20):
    best_loss = float("inf")
    stopping_counter = 0
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = sum(criterion(model(batch_X), batch_Y).item() for batch_X, batch_Y in train_loader) / len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        test_loss = sum(criterion(model(batch_X), batch_Y).item() for batch_X, batch_Y in test_loader) / len(test_loader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        if test_loss < best_loss:
            best_loss, stopping_counter = test_loss, 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered")
                break
    return train_losses, test_losses

def evaluate_model(model, X_test_tensor, Y_test, scaler_y):
    y_pred = scaler_y.inverse_transform(model(X_test_tensor).cpu().detach().numpy())
    Y_test_actual = scaler_y.inverse_transform(Y_test)
    rmse = [np.sqrt(mean_squared_error(Y_test_actual[:, i], y_pred[:, i])) for i in range(Y_test_actual.shape[1])]
    print(f"RMSE - Teff: {rmse[0]:.4f}, logg: {rmse[1]:.4f}, [Fe/H]: {rmse[2]:.4f}")

def main():

    # 1. 加载数据集 第一列为label标签，其余列为特征列
    data = pd.read_csv("output.csv")
    labels = data["Type"].values
    drop_columns = ["Type", "Teff", "logg", "[Fe/H]"]
    X = data.drop(columns=drop_columns).values
    Y = data[["Teff", "logg", "[Fe/H]"]].values

    # 2. 开启gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 数据编码和向量转换
    X_scaled, Y_scaled, scaler_x, scaler_y = preprocess_data(X, Y)
    X_train, X_test, Y_train, Y_test, labels_train, labels_test = split_data(X_scaled, Y_scaled, labels)
    X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor = prepare_tensors(X_train, X_test, Y_train, Y_test, device)


    train_loader, test_loader = get_data_loaders(X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor)
    model = TransformerModel(X_train.shape[1], 128, Y_train.shape[1]).to(device)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.0001)
    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, device)
    evaluate_model(model, X_test_tensor, Y_test, scaler_y)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')

if __name__ == "__main__":
    main()
