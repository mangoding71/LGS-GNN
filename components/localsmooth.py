
import numpy as np
from scipy.optimize import linprog


def graph_smoothing(G, delta_A_add):
    """
    Perform graph smoothing by computing the average perturbation from neighboring nodes
    and adding it to the original graph.
    """

    num_nodes = G.shape[0]
    delta_A_smooth = np.zeros_like(delta_A_add)

    for i in range(num_nodes):
        neighbors_i = np.nonzero(delta_A_add[i])[0]

        if len(neighbors_i) > 0:
            delta_A_smooth[i] = np.mean(delta_A_add[neighbors_i], axis=0)

    G_smooth = G + delta_A_smooth
    return G_smooth


def local_averaging(G_smooth, theta):
    """
    Perform local averaging by computing the local average feature for each node
    and updating node features.
    """

    num_nodes = G_smooth.shape[0]
    f_local_avg = np.zeros_like(theta)

    for i in range(num_nodes):
        neighbors_i = np.nonzero(G_smooth[i])[0]
        avg_local = (1 / (1 + len(neighbors_i))) * (
            np.sum(theta[i]) + np.sum(theta[neighbors_i])
        )
        f_local_avg[i] = theta[i] + avg_local

    return f_local_avg


