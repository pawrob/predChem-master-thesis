import torch

from model.loss import loss_function, loss_function_old_ocgnn
from utils.plot import plot
from utils.utils import get_radius


def train_new_ocgnn(model, optimizer, train_loader, device, data_center, r_pos, r_neg, nu, e, beta, plot_option, epoch):
    model.train()
    y_true = []
    outputs_arr = torch.zeros(0).to(device)
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        y_true.append(batch.y.cpu().numpy())

        output = model(batch.x.float(), batch.edge_index, batch.batch)
        loss, dist, score = loss_function(nu, data_center, output, r_pos, r_neg, batch.y, beta)

        loss.backward()
        optimizer.step()

        outputs_arr = torch.cat([outputs_arr, output])


    if e % 100 == 0:
        y_true = [element for sublist in y_true for element in sublist]
        plot(outputs_arr, "ocgnn/train/epoch_" + str(e), 2, data_center, r_pos, r_neg, y_true, plot_option=plot_option)

    return loss, model, r_pos

def train_old_ocgnn(model, optimizer, train_loader, device, data_center, radius, r_neg, nu, e, plot_option, epoch):
    model.train()
    y_true = []
    dist_sum = torch.zeros(0).to(device)
    outputs_arr = torch.zeros(0).to(device)
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        y_true.append(batch.y.cpu().numpy())

        output = model(batch.x.float(), batch.edge_index, batch.batch)
        loss, dist, score = loss_function_old_ocgnn(nu, data_center, output, radius)

        loss.backward()
        optimizer.step()

        dist_sum = torch.cat([dist_sum, dist])
        outputs_arr = torch.cat([outputs_arr, output])

    if epoch % 3 == 0:
        radius.data = torch.tensor(get_radius(dist_sum, nu), device=device)

    if e % 100 == 0:
        y_true = [element for sublist in y_true for element in sublist]
        plot(outputs_arr, "ocgnn/train/epoch_" + str(e), 2, data_center, radius, r_neg, y_true, plot_option=plot_option)

    return loss, model, radius

def train_gnn(model, optimizer, criterion, train_loader, device):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
    return loss
