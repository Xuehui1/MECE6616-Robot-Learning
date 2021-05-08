from base import RobotPolicy
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class DNN(torch.nn.Module):
    def __init__(self, input_D):
        super(DNN, self).__init__()
        self.linear1 = torch.nn.Linear(input_D,256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def predict(self, features):
        self.eval()  # Sets network in eval mode (vs training mode)
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


class POSBCRobot(RobotPolicy):
    """ Implement solution for Part1 below """

    def train(self, data):
    
        for key, val in data.items():
            print(key, val.shape)
        print("Using dummy solution for RGBBCRobot")

        obs = data["obs"]
        numx, nx = obs.shape
        X = obs.reshape((numx, nx))
        X = torch.from_numpy(X).float()
        y = data["actions"]
        y = y.reshape(4000, 1)
        y = torch.from_numpy(y).float()

        self.network = DNN(nx)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9) 
        self.criterion = torch.nn.MSELoss()  # loss
        self.num_epochs = 1000
        self.batchsize = 50
        self.shuffle = True

        self.network.train()
        dataset = MyDataset(y, X)
        loader = DataLoader(dataset, batch_size=self.batchsize)

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for i, data in enumerate(loader):
                features = data['feature'].float()
                labels = data['label'].float()
                self.optimizer.zero_grad()
                predictions = self.network(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
            print ('loss', total_loss / i)

    def get_action(self, obs):
        self.network.eval()
        action=self.network.predict(obs)
        num=action-int(action)
        if num>0.5:
            action=int(action)+1
        else:
            action=int(action)

        return action


if __name__ == '__main__':
    cc=POSBCRobot()

