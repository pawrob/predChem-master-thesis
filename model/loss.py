import numpy as np
import torch


def loss_function(nu, data_center, outputs, r_pos, r_neg, l, beta):
    dist, scores = anomaly_score(data_center, outputs, r_pos, r_neg, l.squeeze(), beta)
    loss = torch.mean(torch.max(torch.zeros_like(scores), scores))

    return loss, dist, scores

def loss_function_old_ocgnn(nu, data_center, outputs, radius):
    dist, scores = anomaly_score_old_ocgnn(data_center, outputs, radius)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss, dist, scores

def anomaly_score_old_ocgnn(data_center, outputs, radius):
    dist = torch.sum((outputs - data_center) ** 2, dim=1)

    scores = dist - radius ** 2
    return dist, scores

def anomaly_score(data_center, outputs, r_pos, r_neg, l, beta):
    dist = torch.pow(torch.norm(outputs - data_center, dim=1), 2)
    scores = (1 - l) * (dist - r_pos ** 2) + l * (r_neg ** 2 - dist) * beta

    return dist, scores


def anomaly_score_evaluate(data_center, outputs, r):
    dist = torch.pow(torch.norm(outputs - data_center, dim=1), 2)
    scores = dist - r ** 2

    return dist, scores


def thresholding(score, threshold):
    anomaly_pred = np.zeros(score.shape[0])
    for i in range(score.shape[0]):
        if score[i] > threshold:
            anomaly_pred[i] = 1
    return anomaly_pred.tolist()
