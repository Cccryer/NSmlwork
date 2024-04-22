# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import captcha_setting
from collections import OrderedDict
from torch.nn import functional as F

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((captcha_setting.IMAGE_WIDTH//8)*(captcha_setting.IMAGE_HEIGHT//8)*512, 1024),nn.Dropout(0.1), nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out) 
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out


import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((captcha_setting.IMAGE_WIDTH//8)*(captcha_setting.IMAGE_HEIGHT//8)*512, 1024),
            nn.Dropout(0.1),
            nn.ReLU())
        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=1, batch_first=True)
        self.rfc = nn.Sequential(
            nn.Linear(128, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.cnn_layer1(x)
        out = self.cnn_layer2(out)
        out = self.cnn_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.unsqueeze(1)  # Add a time dimension
        out, _ = self.lstm(out)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.rfc(out)
        return out
    


