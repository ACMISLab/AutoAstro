#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/8 15:04
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import os

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

# Transformer Model
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
        out = self.fc(x[:, -1, :])
        return out

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out

def make_dir():
    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 要创建的文件夹名称
    folder_name = "prediction_task"

    # 组合完整路径
    folder_path = os.path.join(current_dir, folder_name)

    # 检查文件夹是否已存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_name}' 创建成功，路径为: {folder_path}")
    else:
        print(f"文件夹 '{folder_name}' 已存在，路径为: {folder_path}")

def write_md(text):
    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "prediction_task"
    save_dir = os.path.join(current_dir, folder_name)
    file_path= os.path.join(save_dir, "prediction_result.md")

    # 检查文件是否存在，不存在则创建
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
        print(f"文件 {file_path} 不存在，已创建新文件。")


    # 追加用户输入到文件
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n\n')
    print(f"内容已追加到 {file_path}")

def preprocess_data(X, Y):
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_scaled, Y_scaled = scaler_x.fit_transform(X), scaler_y.fit_transform(Y)
    return X_scaled, Y_scaled, scaler_x, scaler_y

def split_data(X, Y, test_size=0.2):
    return train_test_split(X, Y, test_size=test_size, random_state=42)

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


def select_model(model_type,input_size, hidden_size, output_size, device):
    """ 根据模型类型选择合适的数据加载器和模型 """
    if model_type == "Transformer":
        model = TransformerModel(input_size, hidden_size, output_size).to(device)
    elif model_type == "RNN":
        model = RNNModel(input_size, hidden_size, output_size).to(device)
    elif model_type == "LSTM":
        model = LSTMModel(input_size, hidden_size, output_size).to(device)
    elif model_type == "GRU":
        model = GRUModel(input_size, hidden_size, output_size).to(device)
    else:
        raise ValueError("Unsupported model type! Choose from ['LSTM', 'Transformer', 'RNN', 'GRU']")

    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=300, patience=30):

    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "prediction_task"
    save_dir = os.path.join(current_dir, folder_name)

    best_loss = float("inf")
    stopping_counter = 0
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)


        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)


        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss, stopping_counter = test_loss, 0
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered")
                break

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_curve.png')
    plt.close()

def test_model(model, X_test_tensor, Y_test, scaler_y, label_name):
    y_pred = scaler_y.inverse_transform(model(X_test_tensor).cpu().detach().numpy())
    Y_test_actual = scaler_y.inverse_transform(Y_test)
    rmse = [np.sqrt(mean_squared_error(Y_test_actual[:, i], y_pred[:, i])) for i in range(Y_test_actual.shape[1])]

    text=f"Testing Result:\n"
    for index,sub_rmse in enumerate(rmse):
        text+=f"RMSE - {label_name[index]}: {sub_rmse:.4f};"

    # write_md(text=f"Testing Result:\nRMSE - Teff: {rmse[0]:.4f}, logg: {rmse[1]:.4f}, [Fe/H]: {rmse[2]:.4f}")
    # print(f"Testing Result:\n RMSE - Teff: {rmse[0]:.4f}, logg: {rmse[1]:.4f}, [Fe/H]: {rmse[2]:.4f}")

    write_md(text=text)
    print(text)

def adjust_learning_rate(base_lr=0.0001, batch_size=4096, base_batch_size=4096):
    return base_lr * (batch_size / base_batch_size)



make_dir() #Create a directory without making any modifications

# 1. Load dataset
data = pd.read_csv("/ai/astro/AutoAstro/Data/ts_data/plasticc.csv")
# Select relevant columns for features and label
features = data[['host_photoz', 'host_photoz_error'] + [f'fulx_{i}' for i in range(28)]].values
label_name = ["host_specz"]
labels = data[label_name].values

# 2. Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Data encoding and vector conversion
X_scaled, Y_scaled, scaler_x, scaler_y = preprocess_data(features, labels) #Do not make any modifications
X_train, X_test, Y_train, Y_test= split_data(X_scaled, Y_scaled) #Do not make any modifications
X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor = prepare_tensors(X_train, X_test, Y_train, Y_test, device)

# 4. Training configuration
batch_size = 4096 # Default batch size is 4096
model_type = "LSTM" # Can be modified to "GRU", "Transformer", or "RNN"
epochs=300
patience=30
learning_rate=adjust_learning_rate(batch_size=batch_size) #Do not make any modifications

write_md(text=f"Training Configuration: Batch Size: {batch_size}, Model Type: {model_type}, Epochs: {epochs}, Early Stopping Patience: {patience}, Learning rate: {learning_rate}") # Log training configuration
train_loader, test_loader = get_data_loaders(X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, batch_size=batch_size)
model = select_model(model_type=model_type,input_size=X_train.shape[1], hidden_size=128, output_size=Y_train.shape[1],device=device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Model training and testing
train_model(model, train_loader, test_loader, criterion, optimizer,epochs=epochs,patience=patience)
test_model(model, X_test_tensor, Y_test, scaler_y, label_name)