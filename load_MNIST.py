import os
import numpy as np
import struct
from torch.utils.data import Dataset
import torch.nn as nn



ROOT = './MNIST'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'

def load_images(training = True):
    if training:
        images_path = os.path.join(ROOT, train_images)
    else:
        images_path = os.path.join(ROOT, test_images)

    with open(images_path, 'rb') as f:
        data = f.read(16)
        des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
        train_x = np.zeros((img_nums, row * col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                train_x[index, :] = np.array(struct.unpack_from('>' + 'B' * (row * col), data, 0)).reshape(1, 784)
        f.close()
    return train_x

def load_labels(training = True):
    if training:
        labels_path = os.path.join(ROOT, train_labels)
    else:
        labels_path = os.path.join(ROOT, test_labels)
    with open(labels_path, 'rb') as f:
        data = f.read(8)
        des,label_nums = struct.unpack_from('>II', data, 0)
        train_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            train_y[index,:] = np.array(struct.unpack_from('>B', data, 0)).reshape(1,1)
        f.close()

    return train_y

class MNISTDataset(Dataset):
    def __init__(self, training=True):
        super(MNISTDataset, self).__init__()
        self.data = load_images(training)
        self.labels = load_labels(training)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

