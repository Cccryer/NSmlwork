import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
original_columns = ['Protocol', 'Flow Duration', 'Total Fwd Packets',
       'Total Backward Packets', 'Total Length of Fwd Packets',
       'Total Length of Bwd Packets', 'Fwd Packet Length Max',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Bwd Packet Length Max',
       'Bwd Packet Length Min', 'Bwd Packet Length Mean',
       'Bwd Packet Length Std', 'Flow IAT Mean', 'Flow IAT Std',
       'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
       'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
       'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
       'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
       'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
       'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
       'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
       'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
       'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
       'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
       'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
       'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
       'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
       'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
       'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
       'Inbound']

path1 = './dataset/cicddos2019/csv-03-11/03-11/' #for train
path2 = './dataset/cicddos2019/csv-01-12/01-12/' #for test
frac = 0.01
fracsize = str(frac*100)


df_train = pd.read_csv(path2 + fracsize + '_percent_1' + '.csv', low_memory=False)
df_test = pd.read_csv(path1 + fracsize + '_percent_1' + '.csv', low_memory=False)

df_train.drop(columns = 'Destination Port', inplace=True)
df_test.drop(columns = 'Destination Port', inplace=True)

#相关性
# corr = df_train.corr(numeric_only=True).round(2)
# f, ax = plt.subplots(figsize=(40, 40))
# mask = np.triu(np.ones_like(corr, dtype=bool))
# cmap = sns.diverging_palette(200, 30, as_cmap=True)
# sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
# plt.show()
# plt.savefig('./test1')
# print("=================================================================")
# for row in corr.index:
#     for column in corr.columns:
#         if corr.loc[row, column] >=0.8 and row is not column:
#             print(f"{row}, {column}:", corr.loc[row, column])
# print("=================================================================")

#标签统一
df_train.drop(df_train.loc[df_train['Label'] == 'WebDDoS'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'UDPLag'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'DNS'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'SNMP'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'TFTP'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'NTP'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'SSDP'].index, inplace=True)
#df_train.drop(df_train.loc[df_train['Label'] == 'Syn'].index, inplace=True)
#df_train.drop(df_train.loc[df_train['Label'] == 'UDP'].index, inplace=True)


df_test.drop(df_test.loc[df_test['Label'] == 'UDPLag'].index, inplace=True)
df_test.drop(df_test.loc[df_test['Label'] == 'Portmap'].index, inplace=True)
#df_test.drop(df_test.loc[df_test['Label'] == 'Syn'].index, inplace=True)
#df_test.drop(df_test.loc[df_test['Label'] == 'UDP'].index, inplace=True)
df_test.drop(df_test.loc[df_test['Label'] == 'UDPLag'].index, inplace=True)
df_train.drop(df_train.loc[df_train['Label'] == 'UDPLag'].index, inplace=True)



def pie(dataset, flag):
    pie, ax = plt.subplots(figsize=[10,6])
    class_data = dataset['Label'].value_counts()#.sample(frac=1.0)
    print(class_data)

    ax.pie(x=class_data, labels=class_data.keys(), pctdistance=0.4, autopct="%.2f%%")
    ax.set_title("Attack Types in CICDDoS-2019 Day2 Training set", fontdict={'fontsize': 14})
    plt.show()
    if flag == 1:
        plt.savefig('./test2')
    else:
        plt.savefig('./test3')
    plt.show()
# pie(df_train,1)
# pie(df_test,2)

X_train = df_train.drop('Label', axis=1)
Y_train = df_train['Label']


X_test = df_test.drop('Label', axis=1)
Y_test = df_test['Label']


X_train = X_train.astype('float64')
X_test = X_test.astype('float64')


labels1 = set(df_train['Label'].unique())

labels2 = set(df_test['Label'].unique())

labels = labels1.union(labels2)

labels = list(labels)
print("=================================================================")
print('All labels: ', labels)
print('TRAIN', labels1)
print('TEST', labels2)
print("=================================================================")

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

#标签编码
le = LabelEncoder()

Y_train = le.fit_transform(Y_train.values)
Y_test = le.fit_transform(Y_test.values)




#删除低方差特征
from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train, Y_train)
constant_columns = [column for column in X_train.columns
                if column not in X_train.columns[constant_filter.get_support()]]

# X_train = torch.tensor(constant_filter.transform(X_train), dtype=torch.float32)
# X_test = torch.tensor(constant_filter.transform(X_test), dtype=torch.float32)
X_train = constant_filter.transform(X_train)
X_test = constant_filter.transform(X_test)

print("===================Removed columns===============================")
# Printing removed columns
for column in constant_columns:
    print("Removed ", column)
print("=================================================================")

#归一化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = np.array(scaler.fit_transform(X_train))
X_test = np.array(scaler.transform(X_test))


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

Y_train = torch.tensor(Y_train, dtype=int)
Y_test = torch.tensor(Y_test, dtype=int)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda")
# 定义自动编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
     
        lstm_out, _ = self.lstm(x)
     
        last_output = lstm_out[:, -1]
        out = self.fc(lstm_out)
        return out

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (input_size // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 在第二个维度上增加一个维度，因为卷积层的输入需要有通道维度
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 展平特征图
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义特征提取函数
def feature_extraction(X_train_mm, input_dim, encoding_dim):
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 15
    batch_size = 2048

    train_dataset = TensorDataset(X_train_mm, X_train_mm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return autoencoder.encoder

# 构建多类分类模型
def multiClassModel(input_dim, output_dim):
    model = LSTMClassifier(input_dim, 50, output_dim).to(device)
    #model = CNN(input_size=input_dim, num_classes=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

# 特征提取
# encoder = feature_extraction(X_train, X_train.shape[1], 8)

X_train_mm_extract = X_train
X_test_mm_extract = X_test
print("=================================", X_train_mm_extract.shape)
print("=================================", X_test_mm_extract.shape)
# print(X_test_mm_extract.shape)
# print(X_train_mm_extract.shape)
# 模型构建
model, criterion, optimizer = multiClassModel(X_train_mm_extract.shape[1], len(torch.unique(Y_train)))

model.to(device)
criterion.to(device)

# 模型训练
num_epochs = 10
batch_size = 8192


train_dataset = TensorDataset(X_train_mm_extract, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
import time
strat_time = time.time()
i = 0
for epoch in range(num_epochs):
    model.train()
    print("epoch : ", i, '\n')
    i = i + 1
    for data in train_loader:
        inputs, targets = data
        print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print("epoch : ", i, " loss is :" , loss.item(), '\n')
        loss.backward(retain_graph=True)
        optimizer.step()
end_time = time.time()
print("runing time is ", end_time - strat_time)

history = {'loss': []}
model.eval()
with torch.no_grad():
    for data in train_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        history['loss'].append(loss.item())

plt.plot(history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim(0, num_epochs-1)
plt.show()
plt.savefig('./loss')
from sklearn.metrics import confusion_matrix


def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    all_labels = []
    all_pred = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))
    cm = confusion_matrix(all_labels, all_pred)
    return cm

# 评估模型准确率
test_dataset = TensorDataset(X_test_mm_extract, Y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

cm = evaluate_model(model, test_loader)
plt.figure(figsize=(15, 15))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.savefig('./cm')




