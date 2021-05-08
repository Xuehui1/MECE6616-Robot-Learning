from base import RobotPolicy
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.dropout(x,p=0.2)
        return x

    def predict(self, features):
        self.eval()
        features = torch.from_numpy(features).float()
        return self.forward(features).detach().numpy()

class MyDataset(Dataset):
    def __init__(self, labels, features):
        super(MyDataset, self).__init__()
        self.labels = labels
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}


class RGBBCRobot2(RobotPolicy):
    """ Implement solution for Part3 below """

    def train(self, data):
        print('i am here')
        for key, val in data.items():
            print(key, val.shape)
        print("Using dummy solution for RGBBCRobot2")
        pass

        obs = data["obs"]
        X = np.swapaxes(obs, 1, 3)
        X = torch.from_numpy(X).float()
        y = data["actions"]
        y = y.reshape(12000, 1)
        y = torch.from_numpy(y).long()

        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        dataset = MyDataset(y, X)
        trainloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)

        for epoch in range(30):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs = data['feature']
                labels = data['label']
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.squeeze(1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('loss', running_loss / (i + 1))

    def get_action(self, obs):
        self.net.eval()
        obs = obs.reshape(1, 64, 64, 3)
        X = np.swapaxes(obs, 1, 3)
        X = torch.from_numpy(X).float()
        outputs = self.net(X)
        _, predicted = torch.max(outputs, 1)
        return predicted[0]

