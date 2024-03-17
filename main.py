import argparse
import warnings

import torch
from matplotlib import pyplot as plt
from torch.nn import BCELoss
from torch.optim import Adam
from torch_geometric.nn.summary import summary
import random
from dataloader.dataset import load_dataset, load_dataset_for_binary_classification
from model.GNN import Embedder, Classifier
from model.evaluator import evaluate_ocgnn, evaluate_gnn
from model.trainer import train_new_ocgnn, train_gnn, train_old_ocgnn
from utils.utils import init_center, read_csv_to_list, to_molecule, draw_molecule, explain, aggregate_edge_directions

warnings.filterwarnings("ignore")

epochs = 51
batch_size = 1

lr = 0.0005
nu = 0.1
eps = 0.01
beta = 10



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.csv:
        csv_data = read_csv_to_list("experiments/new_ocgnn_v1.csv")
        iterations = len(csv_data)
    else:
        iterations = 1
    for i in range(iterations):
        if args.csv==True:
            print(csv_data[i])
            layers = int(csv_data[i][0])
            output_dim = int(csv_data[i][1])
            layers_type = str(csv_data[i][2])
            is_bias = bool(csv_data[i][3])
            dataset_combination = str(csv_data[i][4])
            weight_decay = float(csv_data[i][5])
            args.network = str(csv_data[i][6])
        else:
            print("Running with default values")
            layers = 4
            output_dim = 2
            layers_type = 'SAGE'  # 'SAGE' or 'GAT'
            is_bias = False
            dataset_combination = '[0.21, 0.79]'
            weight_decay = 0.0005
        if args.network == 'new_ocgnn':
            train_loader, test_loader, num_features = load_dataset(batch_size,dataset_combination)
            batch = next(iter(train_loader))
            mol = to_molecule(batch[i])
            draw_molecule(mol)

            model = Embedder(num_features, output_dim, layers, layers_type, is_bias).to(device)

            batch = next(iter(train_loader))
            # print(summary(model, batch.x.float(), batch.edge_index, batch.batch))

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            data_center = init_center(train_loader, model, device, batch_size, eps)
            r_pos = torch.tensor(1, device=device)
            r_neg = torch.tensor(1, device=device)
        elif args.network == 'ocgnn':
            train_loader, test_loader, num_features = load_dataset(batch_size,dataset_combination)
            model = Embedder(num_features, output_dim, layers, layers_type, is_bias).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            data_center = init_center(train_loader, model, device, batch_size, eps)
            radius = torch.tensor(0, device=device)
        else:
            train_loader, test_loader, num_features = load_dataset_for_binary_classification(batch_size,dataset_combination)
            model = Classifier(num_features, output_dim, 1, layers,is_bias).to(device)

            optimizer = Adam(model.parameters(), lr=0.0001)
            criterion = BCELoss()

        for epoch in range(epochs):
            if args.network == 'new_ocgnn':
                loss, model, radius = train_new_ocgnn(model, optimizer, train_loader, device, data_center, r_pos, r_neg, nu,
                                                      epoch, beta, args.plot, epochs)
                auc, acc, prec, rec, f1 = evaluate_ocgnn(epoch, model, test_loader, device, data_center, r_pos, r_neg,
                                                         args.plot)
            elif args.network == 'ocgnn':
                loss, model, radius = train_old_ocgnn(model, optimizer, train_loader, device, data_center, radius,
                                                      torch.tensor(0, device=device), nu, epoch, args.plot, epochs)
                auc, acc, prec, rec, f1 = evaluate_ocgnn(epoch, model, test_loader, device, data_center, radius,
                                                         torch.tensor(0, device=device), args.plot)
            else:
                loss = train_gnn(model, optimizer, criterion, train_loader, device)
                auc, acc, prec, rec, f1 = evaluate_gnn(model, test_loader, device, epoch, args.plot)
            if epoch % 10 == 0:
                print(
                    'Epoch {:03d}, Loss: {:.9f} AUC: {:.4f} | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(
                        epoch, loss.item(), auc,
                        acc, prec, rec, f1))

        test_batch = next(iter(test_loader))
        mol = to_molecule(test_batch)

        # for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
        edge_mask = explain('ig', test_batch,model, target=0)
        edge_mask_dict = aggregate_edge_directions(edge_mask, test_batch)
        plt.figure(figsize=(10, 5))
        # plt.title(title)
        draw_molecule(mol, edge_mask_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Positive GNN')
    parser.add_argument("--network", type=str, default='new_ocgnn',
                        help="type of used network new_ocgnn/ocgnn/binary_classification")
    parser.add_argument("--plot", type=str, default='save',
                        help="Whether to plot the results or not save, save/show")
    parser.add_argument("--csv", type=bool, default=False,
                        help="read parameters from csv file or not")
    args = parser.parse_args()

    main(args)
