import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.decomposition import PCA


def plot(tensor, title, dim, data_center=None,radius_pos=None, radius_neg=None, labels=None, plot_option='save'):
    if not os.path.exists('data/plots/ocgnn/train'):
        os.makedirs('data/plots/ocgnn/train')
    if not os.path.exists('data/plots/ocgnn/test'):
        os.makedirs('data/plots/ocgnn/test')
    if not os.path.exists('data/plots/binary/test'):
        os.makedirs('data/plots/binary/test')

    array = tensor.cpu().detach().numpy()
    # pca = PCA(n_components=2)
    # pca.fit(tensor.cpu().detach().numpy())
    # array = pca.transform(tensor.cpu().detach().numpy())

    X = array[:, 0]
    if dim != 0:
        Y = array[:, 1]
    else:
        Y = np.zeros(X.shape[0])

    colors = labels
    plt.scatter(X, Y, c=colors, cmap='viridis', s=10)
    if dim == 2:
        center = data_center.cpu().detach().numpy()
        radius_pos = radius_pos.cpu().detach().numpy()
        radius_neg = radius_neg.cpu().detach().numpy()
        circle = Circle(center, radius_pos, edgecolor='black', facecolor='none')
        plt.scatter(center[0], center[1], c='red', label='Circle Center')
        plt.gca().add_patch(circle)
        circle = Circle(center, radius_neg, edgecolor='black', facecolor='none')
        plt.scatter(center[0], center[1], c='red', label='Circle Center')
        plt.gca().add_patch(circle)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.title(title)
    plt.grid(True)

    if plot_option == 'save':
        plt.savefig('data/plots/' + title)
        plt.close()
    elif plot_option == 'show':
        plt.show()
