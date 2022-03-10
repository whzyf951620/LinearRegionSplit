from model_MNIST import SimpleLinearNet, ResLinearNet
from load_MNIST import MNISTDataset
from torch.utils.data import DataLoader
import torch

def train(model, dataloader, criterion, optimizer, device):
    train_acc = 0
    train_loss = 0
    for index, (data, label) in enumerate(dataloader):
        data = data.float().to(device)
        label = label.view(-1).long().to(device)
        _, _, _, logits = model(data)
        loss = criterion(logits, label)
        _, predictions = torch.max(logits, dim=-1)
        train_acc += (torch.eq(predictions, label).sum() / label.shape[0]).item()
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_acc, train_loss, index

def test(model, dataloader, linear = True, device = 'cuda:0'):
    model.eval()
    test_acc = 0

    f1List = []
    f2List = []
    f3List = []

    for index, (data, label) in enumerate(dataloader):
        data = data.float().to(device)
        label = label.view(-1).long().to(device)
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


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCH = 100
    train_dataset = MNISTDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = MNISTDataset(training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # model = SimpleLinearNet(784, 128, 10)
    model = ResLinearNet(784, 128, 10)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    max_testloss = 0
    for epoch in range(EPOCH):
        model.train()
        train_acc, train_loss, index = train(model, train_dataloader, criterion, optimizer, device)
        print('=============Epoch', epoch, ': Train Loss: ', train_loss / (index + 1), 'Train Acc: ',
              train_acc / (index + 1))

        features, mean_test_acc = test(model, test_dataloader, device)

        if max_testloss <= mean_test_acc:
            max_testloss = mean_test_acc
            savefilename = './best_Resmodel_MNIST_ReLU.pth'
            torch.save(model.state_dict(), savefilename)
            # features_Encoding = LinearSplitEncoding(features, type='Htanh')
            # f1Encode, f2Encode, f3Encode = features_Encoding
            # ShowLinearBoundary(f3Encode, test_dataloader, epoch)
            # print('================================================')

        print('=============Epoch', epoch, ': Test Acc: ', mean_test_acc)
