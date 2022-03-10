from model_MNIST import SimpleLinearNet
from Read_MNIST_Visualization import MNIST_2D_plane, MNIST_ellipse
from load_MNIST import MNISTDataset
from torch.utils.data import DataLoader
import torch
from ActivatedPattern_MNIST import ShowLinearBoundary
from utils import LinearSplitEncoding, _count_batch_transition_torch
import argparse

parser = argparse.ArgumentParser(description="your script description")
parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode')  

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

    for index, data in enumerate(dataloader):
        data = data.float().to(device)
        f1, logits = model(data)
        f1List.append(f1)
        # f2List.append(f2)
        # f3List.append(f3)

    if linear:
        f1 = torch.cat(f1List, dim=0)
        # f2 = torch.cat(f2List, dim=0)
        # f3 = torch.cat(f3List, dim=0)
        # return [f1, f2, f3]
        return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--number', '-n', type = int, help='the number of samples')
    parser.add_argument('--splitnumber', '-sn', type = int, help='the split number of dataset')
    args = parser.parse_args()

    Num = args.number
    split_num = args.splitnumber
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # test_dataset = MNIST_2D_plane(Num = 10000, label_type = 'diff')
    # test_dataset = MNIST_ellipse(Num = Num, label_type = 'diff')
    test_dataset = MNIST_ellipse(Num = Num, label_type = 'diff', split_num = split_num)
    for i in range(100):
        test_dataset.get_data(index=i)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = SimpleLinearNet(784, 128, 10)
        model.to(device)
        savefilename = './ckps/best_model_MNIST_ReLU.pth'
        model.load_state_dict(torch.load(savefilename))

        features = test(model, test_dataloader, device)

        features_Encoding = LinearSplitEncoding(features, type='ReLU')
        # f1Encode, f2Encode, f3Encode = features_Encoding
        f1Encode = features_Encoding

        transitions = _count_batch_transition_torch(f1Encode)
        transitions_count = torch.sum(transitions)
        print(transitions_count.cpu().item())
        # ShowLinearBoundary(f1Encode, test_dataloader)
        print('================================================')
