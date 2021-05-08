from base import RobotPolicy
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
np.random.seed(0)
torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        x = x.view(-1, 16 * 16 * 24)      
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

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

class RGBBCRobot1(RobotPolicy):

    """ Implement solution for Part2 below """
    '''actions (4000,)
        obs (4000, 64, 64, 3)
    '''
    	
	for key, val in data.items():
            print(key, val.shape)
	print("Using dummy solution for RGBBCRobot1")
        
        obs = data["obs"]
        X = np.swapaxes(obs,1,3)
        X = torch.from_numpy(X).float()
        
        Y = data["actions"]
        Y = Y.reshape(4000, 1)
        Y = torch.from_numpy(Y).long()

        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        dataset = MyDataset(Y, X)
        trainloader = DataLoader(dataset,batch_size=6,shuffle=True, num_workers=2)

        for epoch in range(30): 
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs = data['feature']
                labels = data['label']
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs,labels.squeeze(1))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            print('loss', running_loss / (i+1))

    def get_action(self, obs):
        self.net.eval()
        obs=obs.reshape(1,64,64,3)
        X = np.swapaxes(obs, 1, 3)
        X = torch.from_numpy(X).float()
        outputs = self.net(X)
        predicted = torch.max(outputs, 1)
        return predicted[0]


if __name__ == '__main__':
        cc = POSBCRobot()


