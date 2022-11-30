import torch
import os
import numpy.random as nr
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler


num_test_samples_CIFAR10 = [1000] * 10
num_test_samples_CIFAR100 = [100] * 100

DATA_ROOT = '/home/zeh/data'


def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))  # 对1/gamma元素求1/(class_num - 1)次方
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def get_val_test_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples

    val_list = []
    test_list = []
    indices = list(range(0, length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list  # 每个测试集的类选100个作为验证集，生于的900个作为测试集，返回对应的索引


def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0, length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0, length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


def get_oversampled(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building {} CV data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_CIFAR10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_CIFAR100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)

    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
    # class_indices由10个列表组成，第i个列表记录第i类在targets中对应的索引值，所以每一个列表的长度为5000
    # imbal_class_indices 即根据不平衡类的数量在原始数据集（cifar_10）中抽取相应类的数量，从前至后抽取
    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)  # np.hstack将数据集按水平方向拼接一起

    train_cifar.targets = targets[imbal_class_indices]
    train_cifar.data = train_cifar.data[imbal_class_indices]

    assert len(train_cifar.targets) == len(train_cifar.data)

    train_in_idx = get_oversampled_data(train_cifar, num_sample_per_class)
    train_in_loader = DataLoader(train_cifar, batch_size=batch_size,
                                 sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), num_workers=8)
    ds.append(train_in_loader)

    train_instance_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size, shuffle=True)
    ds.append(train_instance_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_loader = DataLoader(test_cifar, batch_size=100, shuffle=False, num_workers=8)

    ds.append(val_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def get_imbalanced(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_CIFAR10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_CIFAR100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=True, transform=TF_train)
    train_in_idx = get_imbalanced_data(train_cifar, num_sample_per_class)  # 返回需要采样样本的索引。
    imbalanced_train_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size, num_workers=8)
    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_in_idx), num_workers=8)
    # SubsetRandomSampler(indices)：无放回地按照给定的索引列表采样样本元素，即第一类采样5000个，第二类采样2997个，第三类采样1077个，以此类推
    ds.append(imbalanced_train_loader)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                                  sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                                  sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds





