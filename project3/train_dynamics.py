import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
np.set_printoptions(suppress=True)
#np.random.seed(0)
#torch.manual_seed(0)


class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    # ---
    # Your code goes here
    def __init__(self, input_D=9, output_D=6):
        super(Net, self).__init__()
        #500 256 128 64
        self.linear1 = nn.Linear(input_D,256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, output_D)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        #x = F.relu(self.linear5(x))
        x = self.linear5(x)
        #x = F.dropout(x,p=0.5)
        return x

    def predict(self, x):
        self.eval()
        return self.forward(x).detach().numpy()

    pass
    # ---


def train(model,train_loader,epoch):
    model.train()
    # ---
    # Your code goes here
    learning_rate=0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()  # loss
    #self.shuffle = True

    total_item = 0
    train_loss = 0.0
    for i, data in enumerate(train_loader,0):
        inputdata, labels = data
        optimizer.zero_grad()
        #.float()
        outputdata = model(inputdata)
        loss = criterion(outputdata, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_item += labels.size(0)
    print ('epoch:',epoch+1,'  train loss: ', train_loss / total_item)


    # ---


def test(model,test_loader):
    model.eval()

    # --
    # Your code goes here
    criterion = nn.MSELoss()

    test_loss = 0.0
    total_item = 0
    for i, data in enumerate(test_loader,0):
        inputdata, labels = data
        outputdata = model(inputdata.float())
        loss = criterion(outputdata.float(), labels.float())

        test_loss += loss.item()
        total_item += labels.size(0)
    
    test_loss=test_loss / total_item
    print ('test loss: ', test_loss)
    # ---
    return test_loss


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    #batch_size=100
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1500)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=1500)

    # ---
    # Your code goes here
    model=Net(input_D=9, output_D=6)
    print('train_dynamics starts training')
    best_loss = 1000000
    best_epoch = -1
    #myepochs=args.epochs
    myepochs=1000
    for epoch in range(myepochs):
        train(model, train_loader, epoch)
        test_loss = test(model, test_loader)

        if test_loss < best_loss:
            print('Best model is here! Again!')
            best_loss = test_loss
            best_epoch = epoch

        model_folder_name = f'epoch_{epoch:04d}_loss_{test_loss:.8f}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')
        
        print('train_dynamics ends training')
        print('Best Epoch: ', best_epoch+1, '  Best Loss :',best_loss)
        print('\n\n')
        
        #if epoch - best_epoch >= 50:
            #print('Stop to avoid overfitting')
            #print('Best Epoch: ', best_epoch, '  Best Loss :',best_loss)
            #break
    # ---


if __name__ == '__main__':
    main()
