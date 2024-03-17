import csv
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.utils import to_networkx
from captum.attr import Saliency, IntegratedGradients
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='red')
    plt.show()


def to_molecule(data):
    ATOM_MAP = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
         'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
         'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La-Lu',
         'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac-Lr',
         'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        # print(data['x'][0])
        data['name'] = ATOM_MAP[data['x'][0]-1]
        del data['x']
    return g





def model_forward(edge_mask, data, model):
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    out = model(data.x.float(), data.edge_index, data.batch)
    print(out)
    return out


def explain(method,  data,model, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data, model),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data, model))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def multiply_dataset(dataset, multiplier):
    dataset_copy = dataset
    for i in range(1, multiplier):
        dataset = ConcatDataset([dataset, dataset_copy])
    return dataset


def split_dataset(dataset, reduction_ratio, maintain_class_balance=True):
    generator = torch.Generator().manual_seed(2137)
    num_samples = len(dataset)
    num_train_samples = int(reduction_ratio[0] * num_samples)
    num_test_samples = num_samples - num_train_samples

    if maintain_class_balance:
        # Collect number of classes
        class_labels, class_counts = torch.unique(dataset.y, return_counts=True)
        num_classes = len(class_labels)
        target_class_counts = (class_counts * reduction_ratio[1]).int()

        filtered_indices = []
        for class_label in range(num_classes):
            # Get indices of samples belonging to the current class
            class_indices = (dataset.y == class_label).nonzero(as_tuple=True)[0]

            # Calculate the number of samples to keep for the current class
            num_samples_to_keep = target_class_counts[class_label]

            # Randomly select samples to keep for the current class
            selected_indices = torch.randperm(len(class_indices), generator=generator)[:num_samples_to_keep]

            # Append the selected indices to the filtered indices list
            filtered_indices.extend(class_indices[selected_indices])

        # Create the filtered dataset by indexing the original dataset
        balanced_dataset = dataset.copy(filtered_indices)

        # Create a new dataset excluding the specified indices
        new_data_list = []
        for idx, data in enumerate(dataset):
            if idx not in filtered_indices:
                new_data_list.append(idx)
        remaining_dataset = dataset.copy(new_data_list)

    else:
        remaining_dataset, balanced_dataset = random_split(dataset, [num_train_samples, num_test_samples])
        remaining_dataset = dataset.copy(remaining_dataset.indices)
        balanced_dataset = dataset.copy(balanced_dataset.indices)

    return remaining_dataset, balanced_dataset


def init_center(loader, model, device, batch_size, eps):
    n_samples = 0
    c = torch.zeros(batch_size, device=device)
    outputs_sum = torch.zeros(0).to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x.float(), batch.edge_index, batch.batch)

            n_samples += outputs.shape[0]
            outputs_sum = torch.cat([outputs_sum, outputs], dim=0)

        c = torch.sum(outputs_sum, dim=0)

    c /= n_samples

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    sqrt = np.sqrt(dist.clone().data.cpu().numpy())
    radius = np.quantile(sqrt, 1 - nu)
    return radius


def read_csv_to_list(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append(row)
    return data
