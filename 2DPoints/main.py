import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from collections import Counter
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
torch.manual_seed(996)

EPOCH = 100
TOTALNUM = 10000

########## For Training ##################
class SimpleLinearNet(nn.Module):
    def __init__(self):
        super(SimpleLinearNet, self).__init__()
        self.layer0 = nn.Linear(2, 4)
        self.relu0 = nn.ReLU()
        self.Htanh0 = nn.Hardtanh()
        self.layer1 = nn.Linear(4, 4)
        self.relu1 = nn.ReLU()
        self.Htanh1 = nn.Hardtanh()
        self.layer2 = nn.Linear(4, 4)
        self.relu2 = nn.ReLU()
        self.Htanh2 = nn.Hardtanh()
        self.classifier = nn.Linear(4, 4)

    def forward(self, x):
        x0 = self.layer0(x)
        x0 = self.relu0(x0)
        x1 = self.layer1(x0)
        x1 = self.relu1(x1)
        x2 = self.layer2(x1)
        x2 = self.relu2(x2)
        y = self.classifier(x2)

        return x0, x1, x2, y

class SimpleDataset(Dataset):
    def __init__(self, TotalNum, SplitRatio = 0.2, training = True):
        super(SimpleDataset, self).__init__()
        self.Num = TotalNum
        self.split = SplitRatio
        self.training = training

        self.data, self.labels = self.generate_random()
        if self.training:
            self.data, self.labels = self.data[:int(self.Num * self.split)], self.labels[:int(self.Num * self.split)]
            # plot2D(self.data, self.labels)
        else:
            self.data, self.labels = self.data[int(self.Num * self.split):], self.labels[int(self.Num * self.split):]
            # plot2D(self.data, self.labels)

    def generate_random(self):
        pp = torch.cat([torch.rand(self.Num, 1), torch.rand(self.Num, 1)], dim=1)
        self.labels = [0 for i in range(self.Num)]
        pn = torch.cat([torch.rand(self.Num, 1), torch.rand(self.Num, 1) * -1], dim=1)
        self.labels += [1 for i in range(self.Num)]
        np = torch.cat([torch.rand(self.Num, 1) * -1, torch.rand(self.Num, 1)], dim=1)
        self.labels += [2 for i in range(self.Num)]
        nn = torch.cat([torch.rand(self.Num, 1) * -1, torch.rand(self.Num, 1) * -1], dim=1)
        self.labels += [3 for i in range(self.Num)]
        self.data = torch.cat([pp, pn, np, nn], dim = 0)
        self.labels = torch.LongTensor(self.labels)

        shuffleList = [i for i in range(self.Num * 4)]
        random.shuffle(shuffleList)
        self.data = self.data[shuffleList]
        self.labels = self.labels[shuffleList]
        return self.data, self.labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]

class SimpleGridDataset(Dataset):
    def __init__(self, TotalNum):
        super(SimpleGridDataset, self).__init__()
        self.Num = TotalNum

        self.data, self.labels = self.generate_grid()

    def generate_grid(self):
        x = torch.linspace(0, 1, int(math.sqrt(self.Num))) * 20
        y = x
        x, y = torch.meshgrid(x, y)
        x, y = x.reshape(-1).unsqueeze(-1), y.reshape(-1).unsqueeze(-1)

        pp = torch.cat([x, y], dim=-1)
        self.labels = [0 for i in range(self.Num)]
        pn = torch.cat([x, y * -1], dim=-1)
        self.labels += [1 for i in range(self.Num)]
        np = torch.cat([x * -1, y], dim=-1)
        self.labels += [2 for i in range(self.Num)]
        nn = torch.cat([x * -1, y * -1], dim=-1)
        self.labels += [3 for i in range(self.Num)]
        self.data = torch.cat([pp, pn, np, nn], dim = 0)
        self.labels = torch.LongTensor(self.labels)

        return self.data, self.labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]
def train(model, dataloader, criterion, optimizer):
    train_acc = 0
    train_loss = 0
    for index, (data, label) in enumerate(train_dataloader):
        _, _, _, logits = model(data)
        loss = criterion(logits, label)
        _, predictions = torch.max(logits, dim=-1)
        train_acc += (torch.eq(predictions, label).sum() / label.shape[0]).item()
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_acc, train_loss, index

def test(model, dataloader, linear = True):
    model.eval()
    test_acc = 0

    f1List = []
    f2List = []
    f3List = []

    for index, (data, label) in enumerate(dataloader):
        data = data.cuda()
        label = label.cuda()
        f1, f2, f3, logits = model(data)
        _, predictions = torch.max(logits, dim=-1)
        test_acc += (torch.eq(predictions, label).sum() / label.shape[0]).item()
        f1List.append(f1)
        f2List.append(f2)
        f3List.append(f3)

    mean_test_acc = test_acc / (index + 1)

    if linear:
        f1 = torch.cat(f1List, dim=0)
        f2 = torch.cat(f2List, dim=0)
        f3 = torch.cat(f3List, dim=0)
        return [f1, f2, f3], mean_test_acc
    else:
        return mean_test_acc

###############################for plot################################

def plot2D(points, labels):
    points = points.numpy()
    labels = labels.numpy()
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, s = 1, c=labels, cmap='coolwarm')
    plt.legend()
    plt.show()

##############################for Getting Pattern and show################
def get_pattern(featureEncode):
    total_unique_pattern = torch.unique(featureEncode, dim=0)
    keylist = [[item, index] for index, item in enumerate(total_unique_pattern)]
    pattern_dict = {}

    for index, item in enumerate(featureEncode):
        for subp, key in keylist:
            if torch.equal(subp, item):
                pattern_dict[index] = key
                continue

    return keylist, pattern_dict

def ShowLinearBoundary(featureEncode, dataloader, epoch = None):
    pattern_list, pattern = get_pattern(featureEncode)
    data2D = []
    for index, (data, labels) in enumerate(dataloader):
        data2D.append(data)

    data2D = torch.cat(data2D, dim=0)
    labels = list(pattern.values())
    PatternCounter = Counter(labels)

    print('Pattern List:', pattern_list)
    print('PatternCounter:', PatternCounter)
    plt.scatter(data2D[:, 0], data2D[:, 1], s = 1, c=labels, cmap='rainbow')
    savefig_path = os.path.join('./patterns', str(epoch) + '.png')
    plt.savefig(savefig_path)
    plt.clf()

def ShowCoutour(activations, cidx, n = 100):
    colors = ['green', 'dodgerblue', 'orange']
    x = torch.range(-n, n - 1).numpy()
    x, y = np.meshgrid(x, x)
    for j, z in enumerate(activations.t()):
        z = z.view(2 * n, 2 * n).cpu().detach().numpy()
        plt.contour(x, y, z, [1e-4], colors=colors[cidx], linewidths=2)
    return

def EncodingReLU(f):
    fEncoding = torch.gt(f, torch.zeros_like(f)).float()
    return fEncoding

def EncodingHardtanh(f):
    n, d = f.shape
    f = f.view(-1)
    positiveIndices = torch.nonzero(torch.gt(f, torch.ones_like(f)))
    f[positiveIndices] = 1
    negativeIndices = torch.nonzero(torch.lt(f, -1 * torch.ones_like(f)))
    f[negativeIndices] = -1
    zeroIndices = torch.nonzero((f > -1) & (f < 1))
    f[zeroIndices] = 0
    f = f.view(n, d)
    return f

def LinearSplitEncoding(features, type = 'ReLU'):
    f1, f2, f3 = features
    if type == 'ReLU':
        Encoding = EncodingReLU
    else:
        Encoding = EncodingHardtanh

    f1Encoding = Encoding(f1)
    f2Encoding = Encoding(f2)
    f3Encoding = Encoding(f3)
    return [f1Encoding, f2Encoding, f3Encoding]


if __name__ == '__main__':

    DRAW_LINEAR = True
    train_dataset = SimpleDataset(TOTALNUM)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = SimpleDataset(TOTALNUM, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = SimpleLinearNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    max_testloss = 0
    '''
    for epoch in range(EPOCH):
        model.train()
        train_acc, train_loss, index = train(model, train_dataloader, criterion, optimizer)
        print('=============Epoch', epoch, ': Train Loss: ', train_loss / (index + 1), 'Train Acc: ', train_acc / (index + 1))


        features, mean_test_acc = test(model, test_dataloader)

        if max_testloss <= mean_test_acc:
            max_testloss = mean_test_acc
            savefilename = './best_model.pth'
            torch.save(model.state_dict(), savefilename)
            # features_Encoding = LinearSplitEncoding(features, type='ReLU')
            # f1Encode, f2Encode, f3Encode = features_Encoding
            # ShowLinearBoundary(f3Encode, test_dataloader, epoch)
            # print('================================================')

        print('=============Epoch', epoch, ': Test Acc: ', mean_test_acc)
    '''
    if DRAW_LINEAR:
        draw_dataset = SimpleGridDataset(TOTALNUM)
        draw_dataloader = DataLoader(draw_dataset, batch_size=256, shuffle=False)
        savefilename = './ckps/best_model.pth'
        # model.load_state_dict(torch.load(savefilename))

        model = model.cuda()
        features, mean_draw_acc = test(model, draw_dataloader)
        # features_Encoding = LinearSplitEncoding(features, type='ReLU')
        # f1Encode, f2Encode, f3Encode = features_Encoding
        # ShowLinearBoundary(f3Encode, draw_dataloader, epoch = 2)
        f1, f2, f3 = features
        ShowCoutour(f1, cidx = 0)
        ShowCoutour(f2, cidx = 1)
        plt.savefig('./patterns/coutour_2DPoints_f12.png')
