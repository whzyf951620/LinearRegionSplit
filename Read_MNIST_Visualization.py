import os
import numpy as np
import struct
import matplotlib.pyplot as plt


ROOT = r'D:\experiments\LinearRegion\MNIST'
train_images = 'train-images.idx3-ubyte'
train_labels = 'train-labels.idx1-ubyte'
test_images = 't10k-images.idx3-ubyte'
test_labels = 't10k-labels.idx1-ubyte'

train_images_path = os.path.join(ROOT, train_images)
with open(train_images_path, 'rb') as f:
    data = f.read(16)
    des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
    train_x = np.zeros((img_nums, row * col))
    for index in range(img_nums):
        data = f.read(784)
        if len(data) == 784:
            train_x[index, :] = np.array(struct.unpack_from('>' + 'B' * (row * col), data, 0)).reshape(1, 784)
    f.close()

train_labels_path = os.path.join(ROOT, train_labels)
with open(train_labels_path, 'rb') as f:
    data = f.read(8)
    des,label_nums = struct.unpack_from('>II', data, 0)
    train_y = np.zeros((label_nums, 1))
    for index in range(label_nums):
        data = f.read(1)
        train_y[index,:] = np.array(struct.unpack_from('>B', data, 0)).reshape(1,1)
    f.close()

label_unique = np.unique(train_y)

tmp = np.full_like(train_y, 5)

for index, label in enumerate(label_unique):
    if index == 4:
        indices, _ = np.nonzero(np.equal(train_y, tmp))

Num = 5

# num1 = np.random.randint(0, 50000)
# num2 = np.random.randint(0, 50000)
# num3 = np.random.randint(0, 50000)

num1, num2, num3 = indices[:3]

img1 = np.expand_dims(train_x[num1], 0)
img2 = np.expand_dims(train_x[num2], 0)
img3 = np.expand_dims(train_x[num3], 0)

A = np.tile(img1, (Num, 1))
B = np.tile(img2, (Num, 1))
C = np.tile(img3, (Num, 1))

s = np.tile(np.expand_dims(np.random.rand(Num), 1), (1, 784))
t = np.tile(np.expand_dims(np.random.rand(Num), 1), (1, 784))

D = s * B + t * C + A

for item in D:
    plt.imshow(item.reshape(28, 28))
    plt.show()
    plt.clf()
