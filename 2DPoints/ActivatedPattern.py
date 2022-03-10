from main import EPOCH, TOTALNUM, SimpleDataset, SimpleLinearNet, \
    train, test, LinearSplitEncoding
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from collections import Counter

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

def ShowLinearBoundary(featureEncode, dataloader):
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
    plt.show()

if __name__ == '__main__':
    ckp_path = './best_model.pth'
    model = SimpleLinearNet()

    test_dataset = SimpleDataset(TOTALNUM, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
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
    features, mean_test_acc = test(model, test_dataloader, linear=True)
    print('--------Mean Test Acc:', mean_test_acc)
    features_Encoding = LinearSplitEncoding(features, type='Htanh')
    f1Encode, f2Encode, f3Encode = features_Encoding

    for item in features_Encoding:
        ShowLinearBoundary(item, test_dataloader)
        print('================================================')

