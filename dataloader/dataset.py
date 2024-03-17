from torch.utils.data import random_split, ConcatDataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from utils.utils import split_dataset


# hiv
# [0.205, 0.795] 15%
# [0.328, 0.672] 10%
# [0.69, 0.31] 5%
# bace
# [0.21, 0.79] - 15%
# [0.135, 0.865] - 10%
# [0.065, 0.935] - 5%

def load_dataset(batch_size,dataset_combination):
    dataset = MoleculeNet(root="./data", name="bace")
    train_dataset, test_dataset = split_dataset(dataset, reduction_ratio=[0.9, 0.1], maintain_class_balance=True)

    labels = train_dataset.y
    positive_samples = train_dataset[labels == 0]
    negative_samples = train_dataset[labels == 1]

    train_positive_samples = positive_samples
    train_negative_samples, test_negative_samples = random_split(negative_samples, eval(dataset_combination))

    # train_negative_samples = multiply_dataset(train_negative_samples, 2)
    combined_train_dataset = ConcatDataset([train_positive_samples, train_negative_samples])
    print("%neg: " + str(len(train_negative_samples)/len(combined_train_dataset)*100))
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, dataset.num_features


def load_dataset_for_binary_classification(batch_size,dataset_combination):
    dataset = MoleculeNet(root="./data", name="bace")
    train_dataset, test_dataset = split_dataset(dataset, reduction_ratio=[0.9, 0.1], maintain_class_balance=True)

    labels = train_dataset.y
    positive_samples = train_dataset[labels == 0]
    negative_samples = train_dataset[labels == 1]

    train_positive_samples = positive_samples
    # [0.21, 0.79] - 15%
    # [0.135, 0.865] - 10%
    # [0.065, 0.935] - 5%
    train_negative_samples, test_negative_samples = random_split(negative_samples, eval(dataset_combination))

    combined_train_dataset = ConcatDataset([train_positive_samples, train_negative_samples])
    print("%neg: " + str(len(train_negative_samples)/len(combined_train_dataset)*100))

    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, dataset.num_features
