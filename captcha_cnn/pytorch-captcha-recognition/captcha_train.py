# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
import matplotlib.pyplot as plt
import captcha_setting
# Hyper Parameters
num_epochs = 40
batch_size = 1024
learning_rate = 0.001

def main():
    device = torch.device('cuda')
    model = CNN().to(device)
    # model = CNN_LSTM().to(device)
    model.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader(batch_size)
    print(train_dataloader)
    loss_history = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            # print(images.shape)
            predict_labels = model(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
                loss_history.append(loss.item())
            if (i+1) % 100 == 0:
                torch.save(model.state_dict(), "./model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(model.state_dict(), "./model.pkl")   #current is model.pkl
    print("save last model")

    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.xlim(0, num_epochs-1)
    plt.show()
    plt.savefig('loss_curve.png')

if __name__ == '__main__':
    main()


