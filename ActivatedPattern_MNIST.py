# from main import LinearSplitEncoding
from model_MNIST import SimpleLinearNet
from main_MNIST import test
from load_MNIST import MNISTDataset
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

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

def ShowCoutour(activations, cidx, n = 100):
    colors = ['green', 'dodgerblue', 'orange']
    x = torch.range(-n / 2, n / 2 - 1).numpy()
    x, y = np.meshgrid(x, x)
    for j, z in enumerate(activations.t()[:10]):
        z = z.view(n, n).cpu().detach().numpy()
        plt.contour(x, y, z, [0], colors=colors[cidx], linewidths=0.1)
    return

def ShowLinearBoundary(featureEncode, dataloader):
    pattern_list, pattern = get_pattern(featureEncode)
    data2D = []
    for index, data in enumerate(dataloader):
        data2D.append(data)

    data2D = torch.cat(data2D, dim=0).numpy()
    pca = PCA(n_components = 2)
    data2D = pca.fit_transform(data2D)
    '''
    data2D_3_mean = np.mean(data2D[:2])
    data2D[:2] = np.full_like(data2D[:2], data2D_3_mean)
    '''
    labels = list(pattern.values())
    PatternCounter = Counter(labels)

    print('Pattern List:', pattern_list)
    print('PatternCounter:', PatternCounter)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data2D[:, 0], data2D[:, 1], data2D[:, 2], s = 1, c=labels, cmap='rainbow')
    '''
    plt.scatter(data2D[:, 0], data2D[:, 1], s = 5, c=labels, cmap='rainbow')
    
    plt.show()

if __name__ == '__main__':
    ckp_path = './ckps/best_model_MNIST_ReLU.pth'
    model = SimpleLinearNet(in_channels=784, hidden_channels=128, out_channels=10)

    test_dataset = MNISTDataset(training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    '''
    features_pre, mean_test_acc_pre = test(model, test_dataloader, linear=True)
    print('--------Mean Test Acc:', mean_test_acc_pre)
    features_Encoding_pre = LinearSplitEncoding(features_pre, type='Htanh')
    f1Encode, f2Encode, f3Encode = features_Encoding_pre

    for item in features_Encoding_pre:
        ShowLinearBoundary(item, test_dataloader)
        print('================================================')
    '''
    model.load_state_dict(torch.load(ckp_path))
    model.cuda()
    features, mean_test_acc = test(model, test_dataloader, linear=True)
    print('--------Mean Test Acc:', mean_test_acc)
    # features_Encoding = LinearSplitEncoding(features, type='ReLU')
    # f1Encode, f2Encode, f3Encode = features_Encoding
    f1, f2, f3 = features

    # for item in features_Encoding:
    #     ShowLinearBoundary(item, test_dataloader)
    #     print('================================================')
    print(f1.shape)
    ShowCoutour(f1, cidx = 0, n = 100)
    plt.savefig('./patterns/coutour_MNIST_f1_test.png')
