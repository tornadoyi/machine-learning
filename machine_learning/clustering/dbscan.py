import numpy as np
import random
from machine_learning import dataset, plot


def dbscan(datas, epsilon, min_count_element):
    m = len(datas)

    # calcuate square distance and neighbous
    dist2 = np.square(datas[:, np.newaxis, :] - datas[np.newaxis, :, :]).sum(axis=2) # m x m
    neighbors = dist2 < epsilon  # m x m
    neighbor_counts = neighbors.sum(axis = 1) # m

    # calculate all cores
    cores = neighbor_counts >= min_count_element # m
    valid_cores = np.ones(m, dtype=np.bool)
    core_indexes = np.where(valid_cores & cores)[0]

    # cluster one by one
    k = 1
    C = np.zeros(m, dtype=np.int)
    visits = np.zeros(m, dtype=np.bool)
    while len(core_indexes) > 0:
        # random pick a core object
        o = core_indexes[np.random.randint(0, len(core_indexes))]
        Q = np.array([o], np.int)
        while len(Q) > 0:
            # calculate new elements in current cluster
            q_neighbors = neighbors[Q]
            q_cores = cores[Q]
            q_core_neighbors = q_neighbors[q_cores]
            _, indexes = np.where(q_core_neighbors)
            Q = indexes[visits[indexes] == False]

            # update visit, valid cores and labels
            visits[Q] = True
            C[Q] = k
            valid_cores[Q] = False
            core_indexes = np.where(valid_cores & cores)[0]

        yield C, k

        # next cluster
        k += 1



def update(f, plt, datas, num_clusters):
    paint = False
    try:
        labels, k = next(f)
        plt.erase()
        plt.set_title('Clusters {}'.format(k))
        plt.paint({
            'point_x': datas[:, 0], 'point_y': datas[:, 1],
            'labels': labels, 'num_clusters': k+1})
        paint = True

    except StopIteration:
        pass

    if paint: plt.show()


datas, labels = dataset.point.area(bounds=(-10, -10, 20, 20), num_area=(2, 2), num_area_points=(5, 10), area_border_ratio=0.3)
f = dbscan(datas, epsilon=5, min_count_element=3)
plt = plot.create_cluster_plot()
plt.connect_event('button_press_event', lambda _: update(f, plt, datas, len(np.unique(labels))))
plt.show({'point_x': datas[:, 0], 'point_y': datas[:, 1]})





