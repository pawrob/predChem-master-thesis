import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score

from model.loss import thresholding, anomaly_score_evaluate
from utils.plot import plot


def evaluate_ocgnn(epoch, model, test_loader, device, data_center, r_pos, r_neg, plot_option):
    model.eval()
    outputs_arr = torch.zeros(0).to(device)
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x.float(), batch.edge_index, batch.batch)

            outputs_arr = torch.cat([outputs_arr, output])
            _, scores = anomaly_score_evaluate(data_center, output, r_pos)

            labels = batch.y.tolist()
            scores = scores.cpu().numpy()

            threshold = 0
            pred = thresholding(scores, threshold)

            y_true.append(labels)
            y_pred.append(pred)

        # flatten list
        y_true = [element for sublist in y_true for element in sublist]
        y_pred = [element for sublist in y_pred for element in sublist]

        auc = 0
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if epoch % 100 == 0:
            plot(outputs_arr, "ocgnn/test/epoch_" + str(epoch), 2, data_center, r_pos, r_neg, y_true,
                 plot_option=plot_option)
    return auc, acc, prec, rec, f1


def evaluate_gnn(model, loader, device, epoch, plot_option):
    model.eval()
    outputs_arr = torch.zeros(0).to(device)
    y_true_arr = torch.zeros(0).to(device)
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in loader:
            batch = batch.to(device)
            output, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
            outputs_arr = torch.cat([outputs_arr, embedding])
            y_true_arr = torch.cat([y_true_arr, batch.y])
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(output.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        auc = 0
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred.round())
        prec = precision_score(y_true, y_pred.round())
        rec = recall_score(y_true, y_pred.round())
        f1 = f1_score(y_true, y_pred.round())

        if epoch % 500 == 0:
            plot(outputs_arr, "binary/test/epoch_" + str(epoch), 1, labels=y_true_arr.cpu().detach().numpy(),
                 plot_option=plot_option)
        return auc, acc, prec, rec, f1
