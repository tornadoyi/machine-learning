import copy
import numpy as np
import random
from machine_learning import dataset, plot


def lvq(datas, labels, learning_rate, max_iters=np.inf, stop_mse=1e-1, batch_iters=None):
    assert datas.ndim == 2
    assert labels.ndim == 1
    assert len(datas) == len(labels)

    if batch_iters is None: batch_iters = len(datas)
    cls_y = np.unique(labels)

    # init proto vector
    p_indexes = [np.random.choice(np.where(labels == cls_y[i])[0]) for i in range(len(cls_y))]
    p_x = datas[p_indexes]
    p_x_align = p_x[np.newaxis, :, :]
    p_y = labels[p_indexes]
    mse = np.inf
    pre_p_x = copy.deepcopy(p_x)

    # train
    i = 0
    while True:
        indexes = random.sample(range(len(datas)), batch_iters)
        batch_x = datas[indexes]
        batch_x_align = batch_x[:, np.newaxis, :]
        batch_y = labels[indexes]

        d = np.sqrt(np.sum(np.square(batch_x_align - p_x_align), axis=2))
        min_p_indexes = np.argmin(d, axis=1)
        batch_p_y = p_y[min_p_indexes]

        cond = batch_y == batch_p_y
        match_indexes = np.where(cond == True)
        mismatch_indexes = np.where(cond == False)

        delta = batch_x - p_x[min_p_indexes]
        lr_mat = np.zeros_like(delta)
        lr_mat[match_indexes] = learning_rate
        lr_mat[mismatch_indexes] = -learning_rate

        p_x[min_p_indexes] = p_x[min_p_indexes] + lr_mat * delta

        # calculate mse
        mse = np.square(p_x - pre_p_x).sum() / len(p_x)

        # check stop
        if i >= max_iters: break
        if mse < stop_mse: break

        # next iterator
        pre_p_x = copy.deepcopy(p_x)
        i += 1


    return i, p_x, p_y, mse



def update(event, plt, num_clusters):
    if event.xdata is None: return

    # inference
    v = np.array([event.xdata, event.ydata])[np.newaxis, :]
    d = np.sqrt(np.sum(np.square(v - p_x), axis=1))
    min_index = np.argmin(d)
    cls = p_y[min_index]

    # paint new point
    plt.set_title('Iterators: {} MSE: {:.2f}'.format(iter, mse))
    plt.paint({
        'point_x': v[:, 0], 'point_y': v[:, 1],
        'labels': [cls], 'num_clusters': num_clusters})
    plt.show()


datas, labels = dataset.point.area(bounds=(-10, -10, 20, 20), num_area=(2, 2), num_area_points=(5, 10), area_border_ratio=0.3)
iter, p_x, p_y, mse = lvq(datas, labels, 0.3)
plt = plot.create_cluster_plot()
plt.connect_event('button_press_event', lambda event: update(event, plt, len(np.unique(labels))))
plt.set_title('Iterators: {} MSE: {:.2f}'.format(iter, mse))
plt.show({
        'point_x': datas[:, 0], 'point_y': datas[:, 1],
        'center_x': p_x[:, 0], 'center_y': p_x[:, 1],
        'labels': labels, 'num_clusters': len(np.unique(labels))})



