
import numpy as np
import random
from machine_learning import dataset, plot

def kmean(datas, num_clusters, max_iter=np.inf, stop_mse=1e-3):
    indexes = random.sample(range(len(datas)), num_clusters)
    centers = datas[indexes]

    x = datas[:, np.newaxis, :]
    c = centers[np.newaxis, :, :]
    pre_cls = np.zeros(len(datas))
    i = 0
    while True:
        # calculate class
        d = np.sqrt(np.sum(np.square(x - c), axis=2))
        cls = np.argmin(d, axis=1)

        # recalculate centers
        for j in range(num_clusters):
            xj = x[cls == j]
            if len(xj) == 0: continue
            c[:, j] = np.sum(xj, axis=0) / len(xj)

        # calculate mse
        mse = np.square(cls - pre_cls).sum() / len(cls)

        yield i, cls, np.squeeze(c, axis=0), mse

        # check stop
        if mse <= stop_mse: break
        if i >= max_iter: break

        # next iterator
        pre_cls = cls
        i += 1


def update(f, plt, datas, num_clusters):
    paint = False
    try:
        iter, cls, centers, mse = next(f)
        plt.erase()
        plt.set_title('Iterators: {} MSE: {:.2f}'.format(iter, mse))
        plt.paint({
            'point_x': datas[:, 0], 'point_y': datas[:, 1],
            'center_x': centers[:, 0], 'center_y': centers[:, 1],
            'labels': cls, 'num_clusters': num_clusters})
        paint = True

    except StopIteration:
        pass

    if paint: plt.show()


datas, labels = dataset.point.area(bounds=(-10, -10, 20, 20), num_area=(2, 2), num_area_points=(5, 10), area_border_ratio=0.3)
f = kmean(datas, len(np.unique(labels)))
plt = plot.create_cluster_plot()
plt.connect_event('button_press_event', lambda _: update(f, plt, datas, len(np.unique(labels))))
plt.show({'point_x': datas[:, 0], 'point_y': datas[:, 1]})

