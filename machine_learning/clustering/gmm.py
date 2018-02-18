import numpy as np
import random
import matplotlib.pyplot as plt


def gmm(datas, num_clusters, max_iter, init_mu=None, init_sigma=None, init_alpha=None):
    assert datas.ndim == 2

    m = len(datas)
    k = num_clusters
    n_variate = datas.shape[1]
    diagonal_indexes = list(range(n_variate))

    mu = init_mu
    if mu is None:
        indexes = random.sample(range(m), k)
        mu = datas[indexes][np.newaxis, :] # 1 x k x 2

    sigma = init_sigma
    if init_sigma is None:
        sigma = np.repeat((np.eye(n_variate) * 1.0)[np.newaxis, :, :], k, axis=0) # k x 2 x 2

    alpha = init_alpha
    if init_alpha is None:
        alpha = np.array([1/k] * k)[np.newaxis, :]  # 1 x k

    x = datas[:, np.newaxis, :]   # m x 1 x 2
    k_indexes = list(range(k))
    m_indexes = list(range(m))

    w = None
    for i in range(max_iter):
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

        if np.any(np.isnan(sigma)):
            print('sigma')

    return np.argmax(w, axis=1)


# generate datas with n area
def gen_datas(map_bounds, num_area, area_point_range):
    def random_range(shape, min, max):
        assert max > min
        x = np.random.rand(*shape)
        return min + (max - min) * x

    area_width = (map_bounds[2] - map_bounds[0]) /  num_area[0]
    area_height = (map_bounds[3] - map_bounds[1]) / num_area[1]

    datas = []
    labels = []
    for i in range(num_area[0]):
        for j in range(num_area[1]):
            min_x, min_y = map_bounds[0] + area_width * i, map_bounds[1] + area_height * j
            max_x, max_y = min_x + area_width, min_y + area_height
            r = 0.4
            min_x += (max_x - min_x) * r
            max_x -= (max_x - min_x) * r
            min_y += (max_y - min_y) * r
            max_y -= (max_y - min_y) * r
            num_points = np.random.randint(*area_point_range)
            x = random_range((num_points,), min_x, max_x)[:, np.newaxis]
            y = random_range((num_points,), min_y, max_y)[:, np.newaxis]
            datas.append(np.hstack([x,y]))
            labels.append(np.ones(num_points) * (i * num_area[1] + j))

    datas, labels = np.vstack(datas), np.hstack(labels)
    indexes = np.arange(len(datas))
    np.random.shuffle(indexes)

    return datas[indexes], labels[indexes]


def gen_circle_datas(centers, radius, counts):
    assert len(centers) == len(radius)
    assert len(centers) == len(counts)

    def random_range(shape, min, max):
        assert max > min
        x = np.random.rand(*shape)
        return min + (max - min) * x

    datas, labels = [], []
    for i in range(len(centers)):
        o = centers[i]
        r = radius[i]
        c = counts[i]
        p = np.array([r, 0])

        theta = random_range((c, ), 0, 360)
        cos, sin = np.cos(theta)[:, np.newaxis], np.sin(theta)[:, np.newaxis]
        transform = np.hstack([cos, -sin, sin, cos]).reshape([c, 2, 2])
        v = np.tensordot(transform, p[np.newaxis, :, np.newaxis], axes=[[2], [1]])
        p1 = o + np.squeeze(np.squeeze(v, axis=2), axis=2)
        datas.append(p1)
        labels.append(np.array([i] * c))

    datas, labels = np.vstack(datas), np.hstack(labels)
    indexes = np.arange(len(datas))
    np.random.shuffle(indexes)

    return datas[indexes], labels[indexes]


def show(map_bounds, datas, num_clusters, classes, centers, iters, click_callback):
    global figure, axes_list
    try:
        for ax in axes_list: ax.cla()
    except:
        figure, axes_list = plt.subplots(1, 2)


    org_ax, cls_ax = axes_list
    org_ax.axis([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]])
    cls_ax.axis([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]])
    cls_ax.set_title("Iterators: {}".format(iters))

    # register press event
    cid = figure.canvas.mpl_connect('button_press_event', click_callback)

    # draw original datas
    org_ax.plot(datas[:, 0], datas[:, 1], 'o')

    # draw category datas
    if classes is not None:
        for i in range(num_clusters):
            cls_data = datas[classes == i]
            if len(cls_data) == 0: continue
            cls_ax.plot(cls_data[:, 0], cls_data[:, 1], linestyle='None', marker='o')

    if centers is not None:
        cls_ax.plot(centers[:, 0], centers[:, 1], linestyle='None', marker='+', color='k', markersize=10)

    plt.show()


def show_circle(datas, labels):
    global figure, axes_list
    try:
        for ax in axes_list: ax.cla()
    except:
        figure, axes_list = plt.subplots(1, 2)

    org_ax, cls_ax = axes_list
    for i in np.unique(labels):
        cls_data = datas[labels == i]
        if len(cls_data) == 0: continue
        cls_ax.plot(cls_data[:, 0], cls_data[:, 1], linestyle='None', marker='o')

    plt.show()


MAP_BOUNDS = (-20, -20, 20, 20)
NUM_AREA = (2, 2)
AREA_POINTS_RANGE = (50, 60)
MAX_ITERS = 1000
SHOW_PROGRESS = True
NUM_CLUSTERS = NUM_AREA[0] * NUM_AREA[1]

#init_mu = np.array([[-10, -10], [10, 10], [-10, 10], [10, -10]])
init_mu = None


datas, labels = gen_datas(MAP_BOUNDS, NUM_AREA, AREA_POINTS_RANGE)
classes = gmm(datas, NUM_CLUSTERS, MAX_ITERS, init_mu=init_mu)
show(MAP_BOUNDS, datas, NUM_CLUSTERS, classes, None, MAX_ITERS, None)


#datas, labels = gen_circle_datas([(0, 0), (0, 0)], [2, 5], [20, 20])
#classes = gmm(datas, 2, MAX_ITERS)
#show_circle(datas, classes)