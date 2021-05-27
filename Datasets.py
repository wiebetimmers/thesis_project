from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn import datasets
import torchvision
import numpy as np
from torchvision import transforms


class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
            self.targets = digits.target[:1000]
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
            self.targets = digits.target[1000:1350]
        else:
            self.data = digits.data[1350:].astype(np.float32)
            self.targets = digits.target[1350:]
        self.transforms = transforms
        self.target_names = digits.target_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return sample_x, sample_y


class MNIST(Dataset):
    """Complete MNIST dataset."""

    def __init__(self, mode='train', transforms=None):
        data_set_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.1307,), (0.3081,))
                                                    ]))

        data_set_test = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           (0.1307,), (0.3081,))
                                                   ]))
        self.data = []
        self.targets = []

        if mode == 'train':
            for i in range(40000):
                self.data.append(data_set_train[i][0].numpy())
                self.targets.append(data_set_train[i][1])
        elif mode == 'val':
            for i in range(50000, 60000):
                self.data.append(data_set_train[i][0].numpy())
                self.targets.append(data_set_train[i][1])
        else:
            for i in range(len(data_set_test)):
                self.data.append(data_set_test[i][0].numpy())
                self.targets.append(data_set_test[i][1])

        self.targets = np.array(self.targets)
        self.data = np.array(self.data)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return sample_x, sample_y


def get_digits_loaders(batch_size=50):
    # Initiliazing the data loaders for the digits dataset.

    train_data_digits = Digits(mode='train')
    val_data_digits = Digits(mode='val')
    test_data_digits = Digits(mode='test')

    train_loader_digits = DataLoader(train_data_digits, batch_size=batch_size, shuffle=True)
    val_loader_digits = DataLoader(val_data_digits, batch_size=batch_size, shuffle=False)
    test_loader_digits = DataLoader(test_data_digits, batch_size=batch_size, shuffle=False)
    return train_loader_digits, val_loader_digits, test_loader_digits


def get_mnist_loaders(batch_size=50):
    # Load in the MNIST dataset - (takes longer to load than Digits!, 6)
    train_data_mnist = MNIST(mode='train')
    val_data_mnist = MNIST(mode='val')
    test_data_mnist = MNIST(mode='test')

    train_loader_mnist = DataLoader(train_data_mnist, batch_size=batch_size, shuffle=True)
    val_loader_mnist = DataLoader(val_data_mnist, batch_size=batch_size, shuffle=True)
    test_loader_mnist = DataLoader(test_data_mnist, batch_size=batch_size, shuffle=True)
    return train_loader_mnist, val_loader_mnist, test_loader_mnist


# Labels are the same for both digits and mnist
def get_labels():
    train_data_digits = Digits(mode='train')
    labels = train_data_digits.target_names
    return labels
