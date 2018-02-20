import copy
import numpy as np
import random
from machine_learning import dataset, plot


def gmm(datas, num_clusters, max_iter=np.inf,
        stop_mse_mu=1e-3, stop_mse_sigma=1e-3, stop_mse_alpha=1e-3,
        init_mu=None, init_sigma=None, init_alpha=None):

    assert datas.ndim == 2

    m = len(datas)
    k = num_clusters
    n_variate = datas.shape[1]
    diagonal_indexes = list(range(n_variate))

    mu = init_mu
    if mu is None:
        indexes = random.sample(range(m), k)
        mu = datas[indexes] # k x 2
    mu = mu[np.newaxis, :] # 1 x k x 2

    sigma = init_sigma
    if init_sigma is None:
        sigma = np.repeat((np.eye(n_variate) * 1.0)[np.newaxis, :, :], k, axis=0) # k x 2 x 2

    alpha = init_alpha
    if init_alpha is None:
        alpha = np.array([1/k] * k) # k
    alpha = alpha[np.newaxis, :] # 1 x k

    x = datas[:, np.newaxis, :]   # m x 1 x 2
    k_indexes = list(range(k))
    m_indexes = list(range(m))

    w = None
    pre_mu = copy.deepcopy(mu)
    pre_sigma = copy.deepcopy(sigma)
    pre_alpha = copy.deepcopy(alpha)
    i = 0
    while True:
        # multivariate guassian distribution
        x_mu = (x - mu)[:, :, np.newaxis, :] # m x k x 1 x 2
        x_mu_T = np.transpose(x_mu, [0, 1, 3, 2])  # m x k x 2 x 1
        sigma_inv = np.linalg.inv(sigma)
        g_exp = np.tensordot(x_mu, sigma_inv, axes=[[3], [1]])  # m x k x 1 x k x 2
        g_exp = np.squeeze(g_exp, axis=2)[:, k_indexes, k_indexes] # m x k x 2
        g_exp = g_exp[:, :, np.newaxis, :] # m x k x 1 x 2
        g_exp = np.tensordot(g_exp, x_mu_T, axes=[[3], [2]])  # m x k x 1 x m x k x 1
        g_exp = np.squeeze(np.squeeze(g_exp, axis=5), axis=2)  # m x k x m x k
        g_exp = g_exp[m_indexes, :, m_indexes, :]  # m x k x k
        g_exp = g_exp[:, k_indexes, k_indexes] # m x k

        sigma_det = np.linalg.det(sigma)
        g_coef = 1 / (np.power(2 * np.pi, n_variate / 2) * np.sqrt(sigma_det))  # k
        pdf = g_coef[np.newaxis, :] * np.exp(-0.5 * g_exp) # m x k

        # bayes formula
        w_pdf = alpha * pdf     # m x k
        w = w_pdf / np.sum(w_pdf, axis=1, keepdims=True) # m x k

        # clean zero probabilities
        if np.any(w == 0):
            w = (w + 1e-7) / np.sum(w, axis=1, keepdims=True) # m x k

        # update parameters
        w_sum = np.sum(w, axis=0) # k
        alpha = (1 / m) * w_sum[np.newaxis, :]  # 1 x k
        mu = np.sum(w[:, :, np.newaxis] * x, axis=0) / w_sum[:, np.newaxis]  #  (m x k x 1) * (m x 1 x 2) = m x k x 2  sum(m x k x 2) / k x 1 = k x 2
        mu = mu[np.newaxis, :, :]  # 1 x k x 2
        x_mu = (x - mu)[:, :, np.newaxis, :]  # m x k x 1 x 2
        x_mu_T = np.transpose(x_mu, axes=[0, 1, 3, 2])
        sigma = np.tensordot(x_mu_T, x_mu, axes=[[3], [2]]) # m x k x 2 x m x k x 2
        sigma = sigma[:, k_indexes, :, :, k_indexes, :] # k x m x 2 x m x 2
        sigma = sigma[:, m_indexes, :, m_indexes, :] # m x k x 2 x 2
        sigma = np.sum(w[:, :, np.newaxis, np.newaxis] * sigma, axis=0)  # k x 2 x 2
        sigma = sigma / w_sum[:, np.newaxis, np.newaxis] # k x 2 x 2

        # check inverse matrix
        if np.any(np.linalg.det(sigma) == 0):
            sigma[:, diagonal_indexes, diagonal_indexes] += 1e-7

        # calculate mse
        mse_mu = np.square(mu - pre_mu).sum() / k
        mse_sigma = np.square(sigma - pre_sigma).sum() / k
        mse_alpha = np.square(alpha - pre_alpha).sum() / k

        # return result
        yield i, w, np.squeeze(mu, axis=0), sigma, np.squeeze(alpha, axis=0), mse_mu, mse_sigma, mse_alpha

        # check stop
        if i >= max_iter: break
        if mse_mu <= stop_mse_mu and mse_sigma < stop_mse_sigma and mse_alpha < stop_mse_alpha: break

        # next iterator
        pre_mu = copy.deepcopy(mu)
        pre_sigma = copy.deepcopy(sigma)
        pre_alpha = copy.deepcopy(alpha)
        i += 1



def update(f, plt, datas, num_clusters):
    paint = False
    try:
        iter, w, mu, sigma, alpha, mse_mu, mse_sigma, mse_alpha  = next(f)
        labels = np.argmax(w, axis=1)
        probabilities = np.max(w, axis=1).astype(float)
        plt.erase()
        plt.set_title('Iterators: {} MSE: ({:.2f}, {:.2f}, {:.2f})'.format(iter, mse_mu, mse_sigma, mse_alpha))
        plt.paint({
            'point_x': datas[:, 0], 'point_y': datas[:, 1], 'point_alpha': probabilities,
            'center_x': mu[:, 0], 'center_y': mu[:, 1],
            'labels': labels, 'num_clusters': num_clusters})
        paint = True

    except StopIteration:
        pass

    if paint: plt.show()


datas, labels = dataset.point.area(bounds=(-10, -10, 20, 20), num_area=(2, 2), num_area_points=(5, 10), area_border_ratio=0.3)
#datas, labels = dataset.point.circle([[0, 0], [0, 0]], [3, 6], [30])

f = gmm(datas, len(np.unique(labels)))
plt = plot.create_cluster_plot()
plt.connect_event('button_press_event', lambda _: update(f, plt, datas, len(np.unique(labels))))
plt.show({'point_x': datas[:, 0], 'point_y': datas[:, 1]})

