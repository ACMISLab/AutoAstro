#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/27 18:40
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))  # 需要 (seq_len, batch, d_model)
        return self.fc(x[-1, :, :])  # 取最后一个时间步的输出

# RNN 模型
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出

# GRU 模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出

def make_dir():
    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 要创建的文件夹名称
    folder_name = "class_task"

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
    folder_name = "class_task"
    save_dir = os.path.join(current_dir, folder_name)
    file_path= os.path.join(save_dir, "class_result.md")

    # 检查文件是否存在，不存在则创建
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
        print(f"文件 {file_path} 不存在，已创建新文件。")


    # 追加用户输入到文件
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n\n')
    print(f"内容已追加到 {file_path}")

def preprocess_data(features, labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

    return X_train, X_test, y_train, y_test, label_encoder

def convert_to_tensor(X_train, X_test, y_train, y_test):
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

def adjust_learning_rate(base_lr=0.008, batch_size=4096, base_batch_size=4096):
    return base_lr * (batch_size / base_batch_size)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(2)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(X_train, X_test, y_train, y_test, batch_size):
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

class Trans_TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(2)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def Trans_standardize_data(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)  # 计算全局均值
    std = X_train.std(axis=0, keepdims=True) + 1e-8  # 避免除零
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std  # 使用相同均值和标准差
    return X_train, X_test

def Trans_get_data_loaders(X_train, X_test, y_train, y_test, batch_size):
    # 进行标准归一化
    X_train, X_test = Trans_standardize_data(X_train, X_test)

    # 转换为 PyTorch Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)  # Transformer 不需要 unsqueeze
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = Trans_TimeSeriesDataset(X_train, y_train)
    test_dataset = Trans_TimeSeriesDataset(X_test, y_test)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

def select_model_and_loader(model_type, X_train, X_test, y_train, y_test,device,label_encoder,batch_size):
    """ 根据模型类型选择合适的数据加载器和模型 """
    if model_type == "Transformer":
        train_loader, test_loader = Trans_get_data_loaders(X_train, X_test, y_train, y_test, batch_size)
        model = TransformerClassifier(input_size=1, d_model=64, num_heads=4, num_layers=1, num_classes=len(label_encoder.classes_))
    else:
        train_loader, test_loader = get_data_loaders(X_train, X_test, y_train, y_test, batch_size)
        if model_type == "LSTM":
            model = LSTMClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=len(label_encoder.classes_))
        elif model_type == "RNN":
            model = RNNClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=len(label_encoder.classes_))
        elif model_type == "GRU":
            model = GRUClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=len(label_encoder.classes_))
        else:
            raise ValueError("Unsupported model type! Choose from ['LSTM', 'Transformer', 'RNN', 'GRU']")

    return model.to(device), train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=100, patience=10, warmup_epochs=5):
    model.to(device)
    best_acc = 0.0

    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "class_task"
    save_dir = os.path.join(current_dir, folder_name)

    best_model_path = f"{save_dir}/best_model.pth"
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if epoch < warmup_epochs:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.2f}%")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = 100 * correct / total
    # print(classification_report(all_labels, all_preds))
    return total_loss / len(test_loader), accuracy

def test_model(model,test_loader,device,label_encoder):

    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "class_task"
    save_dir = os.path.join(current_dir, folder_name)

    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())

    plot_confusion_matrix(true_labels, predictions, label_encoder)
    write_md("Final Evaluation Report:\n" + classification_report(true_labels, predictions))
    print("Final Evaluation Report:")
    print(classification_report(true_labels, predictions))

def plot_confusion_matrix(true_labels, predictions, label_encoder):
    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "class_task"
    save_dir = os.path.join(current_dir, folder_name)

    conf_matrix = confusion_matrix(true_labels, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{save_dir}/Confusion.png")
    plt.close()

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):

    # 获取当前执行文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "class_task"
    save_dir = os.path.join(current_dir, folder_name)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid()

    plt.savefig(f"{save_dir}/train_test.png")
    plt.close()




make_dir() #创建目录，不要做任何修改

# 1. 加载数据集 第一列为label标签，其余列为特征列
data = pd.read_csv("/ai/astro/AutoAstro/Data/ts_data/Loamst_Class.csv")
labels = data.iloc[:, 0].values
features = data.iloc[:, 1:].values

# 2.开启gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3.数据编码和向量转换
X_train, X_test, y_train, y_test, label_encoder = preprocess_data(features, labels)
X_train, X_test, y_train, y_test = convert_to_tensor(X_train, X_test, y_train, y_test)

# 4.训练配置
batch_size=4096 # 默认batch_size大小微4096，若使用Transformer模型，则建议设置为1024
model_type = "LSTM"  # 可修改为 "GRU", "Transformer", "RNN"
epochs=100
patience=10
warmup_epochs = 5
learning_rate=adjust_learning_rate(batch_size=batch_size)
write_md(text=f"Training Configuration:\nBatch Size: {batch_size}, Model Type: {model_type}, Epochs: {epochs}, Early Stopping Patience: {patience}, Warmup epochs: {warmup_epochs}, Learning rate: {learning_rate}")  # 训练配置记录
model, train_loader, test_loader = select_model_and_loader(model_type, X_train, X_test, y_train, y_test, device,label_encoder, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5.模型训练与测试
train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs, patience=patience, warmup_epochs=warmup_epochs)
test_model(model, test_loader, device, label_encoder)



# def main():
#
#     make_dir() #创建目录，不要做任何修改
#
#     # 1. 加载数据集 第一列为label标签，其余列为特征列
#     data = pd.read_csv("Data.csv")
#     labels = data.iloc[:, 0].values
#     features = data.iloc[:, 1:].values
#
#     # 2.开启gpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 3.数据编码和向量转换
#     X_train, X_test, y_train, y_test, label_encoder = preprocess_data(features, labels)
#     X_train, X_test, y_train, y_test = convert_to_tensor(X_train, X_test, y_train, y_test)
#
#     # 4.训练配置
#     batch_size=4096 # 默认batch_size大小微4096，若使用Transformer模型，则建议设置为1024
#     model_type = "LSTM"  # 可修改为 "GRU", "Transformer", "RNN"
#     epochs=100
#     patience=10
#     warmup_epochs = 5
#     learning_rate=adjust_learning_rate(batch_size=batch_size)
#     write_md(text=f"Training Configuration:\nBatch Size: {batch_size}, Model Type: {model_type}, Epochs: {epochs}, Early Stopping Patience: {patience}, Warmup epochs: {warmup_epochs}, Learning rate: {learning_rate}")  # 训练配置记录
#     model, train_loader, test_loader = select_model_and_loader(model_type, X_train, X_test, y_train, y_test, device,label_encoder, batch_size)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     # 5.模型训练与测试
#     train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs, patience=patience, warmup_epochs=warmup_epochs)
#     test_model(model, test_loader, device, label_encoder)


# if __name__ == "__main__":
#     main()
