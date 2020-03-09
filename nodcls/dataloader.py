import os
import os.path

import numpy as np
import torch.utils.data as data


class lunanod(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, npypath, fnamelst, labellst, featlst, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.train_data.append(np.load(file))
                self.train_labels.append(label)

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((len(fnamelst), 32, 32, 32))
            self.train_len = len(fnamelst)
        else:
            self.test_data = []
            self.test_labels = []
            self.test_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.test_data.append(np.load(file))
                self.test_labels.append(label)

            self.test_data = np.concatenate(self.test_data)
            self.test_data = self.test_data.reshape((len(fnamelst), 32, 32, 32))
            self.test_len = len(fnamelst)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, feat = self.train_data[index], self.train_labels[index], self.train_feat[index]
        else:
            img, target, feat = self.test_data[index], self.test_labels[index], self.test_feat[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, feat

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len
