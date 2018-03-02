import numpy as np
from sklearn.decomposition import FastICA


def load_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


SAMPLE_NUM = 60000
IMG_SIZE = 28
kernel_size = 15
kernel_num = kernel_size * kernel_size
stride = 1
output_row = int((IMG_SIZE - kernel_size) / stride) + 1
output_col = int((IMG_SIZE - kernel_size) / stride) + 1
channel_num = 32

gridid = 0
idmatrix = np.zeros((IMG_SIZE, IMG_SIZE), dtype=int)
for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
        idmatrix[i, j] = gridid
        gridid = gridid + 1

X_train, y_train = load_mnist('.', kind='train')
print(X_train.shape)

tot = 0
rng = np.random.RandomState(42)
ica = FastICA(random_state=rng, whiten=True, n_components=channel_num,
              max_iter=2000)
weight = np.zeros((output_row * output_col, kernel_num, channel_num))

for i in range(output_row):
    for j in range(output_col):
        print("Window no. %d begin creating patch" % tot)
        window_ul_row = i * stride
        window_ul_col = j * stride
        patch_data = np.zeros((SAMPLE_NUM, kernel_num))

        for s in range(SAMPLE_NUM):
            pixelid = 0
            for row in range(window_ul_row, window_ul_row + kernel_size):
                for col in range(window_ul_col, window_ul_col + kernel_size):
                    patch_data[s, pixelid] = X_train[s, idmatrix[row, col]]
                    pixelid += 1
        print("Window no. %d finished creating patch" % tot)

        print("Window no. %d begin ICA" % tot)
        patch_ICA = ica.fit(patch_data).transform(patch_data)
        print("Window no. %d finished ICA" % tot)
        basis = ica.mixing_
        if basis.min() >= 0:
            basis = basis - np.mean(basis)
        basis = basis / np.max(abs(basis))
        weight[tot] = basis
        print("Window no. %d saved normalized weight" % tot)
        tot += 1

# print("weight shape %d" % weight.shape)
filename = "pretrainW_ksize_" + str(kernel_size) + "_stride_" + str(stride) + \
           "_channel_" + str(channel_num)
np.save(filename, weight)
print("pretrain weight saved")
