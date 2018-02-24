import numpy as np
import random
from machine_learning import dataset, plot


def dist2_min(d): return d.min()

def dist2_max(d): return d.max()

def dist2_avg(d): return d.sum() / (d.shape[0] * d.shape[1])

def agnes(datas, num_clusters, dist_functor=dist2_avg):
    k = num_clusters
    m = len(datas)
    indexes_m = range(m)

    dist = np.square(datas[:, np.newaxis, :] - datas[np.newaxis, :, :]).sum(axis=2)
    clusters = np.arange(m, dtype=np.int)
    clusters_dist = dist.copy()
    clusters_dist[indexes_m, indexes_m] = np.inf

    q = m
    i = 0
    while q > k or clusters_dist.min() is np.inf:
        r, c = np.unravel_index(clusters_dist.argmin(), clusters_dist.shape)
        c1, c2 = clusters[r], clusters[c]

        # merge c2 to c1
        clusters[clusters == c2] = c1

        # clear c2 distance
        clusters_dist[:, c2] = np.inf
        clusters_dist[c2, :] = np.inf

        # update distance between c1 with ci
        dist_c1 = dist[clusters == c1]
        c_indexes = np.unique(clusters)
        for ci in c_indexes:
            if ci == c1: continue
            data_indexes_ci = np.where(clusters == ci)[0]
            dist_c1_ci = dist_c1[:, data_indexes_ci]
            d = dist_functor(dist_c1_ci)
            clusters_dist[[ci, c1], [c1, ci]] = d

        # return
        yield i, clusters

        # next iterator
        q = len(c_indexes)
        i += 1



def update(f, plt, datas):
    paint = False
    try:
        iters, labels = next(f)
        plt.erase()
        plt.set_title('Iterator: {} Clusters: {}'.format(iters, len(np.unique(labels))))
        plt.paint({
            'point_x': datas[:, 0], 'point_y': datas[:, 1],
            'labels': labels, 'num_clusters': len(datas)})
        paint = True

    except StopIteration:
        pass

    if paint: plt.show()


datas, labels = dataset.point.area(bounds=(-10, -10, 20, 20), num_area=(2, 2), num_area_points=(3, 4), area_border_ratio=0.3)
f = agnes(datas, num_clusters=len(np.unique(labels)))
plt = plot.create_cluster_plot()
plt.connect_event('button_press_event', lambda _: update(f, plt, datas))
plt.show({'point_x': datas[:, 0], 'point_y': datas[:, 1]})

