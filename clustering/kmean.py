
import numpy as np
import random
import matplotlib.pyplot as plt

def kmean(datas, num_clusters, max_iter):
    indexes = random.sample(range(len(datas)), num_clusters)
    centers = datas[indexes]

    x = datas[:, np.newaxis, :]
    c = centers[np.newaxis, :, :]
    cls = None

    for i in range(max_iter):
        # calculate class
        d = np.sqrt(np.sum(np.square(x - c), axis=2))
        cls = np.argmin(d, axis=1)

        # recalculate centers
        for j in range(num_clusters):
            xj = x[cls == j]
            if len(xj) == 0: continue
            c[:, j] = np.sum(xj, axis=0) / len(xj)

    return cls, np.squeeze(c, axis=0)


def random_range(shape, min, max):
    assert max > min
    x = np.random.rand(*shape)
    return min + (max - min) * x






# generate datas with 4 area
def gen_datas(NUM_AREA_POINTS = 10):
    minv, maxv = 0, 10

    x = random_range((NUM_AREA_POINTS,), -maxv, -minv)[:, np.newaxis]
    y = random_range((NUM_AREA_POINTS,), minv, maxv)[:, np.newaxis]
    top_left = np.hstack([x,y])

    x = random_range((NUM_AREA_POINTS,), minv, maxv)[:, np.newaxis]
    y = random_range((NUM_AREA_POINTS,), minv, maxv)[:, np.newaxis]
    top_right = np.hstack([x,y])

    x = random_range((NUM_AREA_POINTS,), minv, maxv)[:, np.newaxis]
    y = random_range((NUM_AREA_POINTS,), -maxv, -3)[:, np.newaxis]
    bottom_right = np.hstack([x,y])

    x = random_range((NUM_AREA_POINTS,), -maxv, -minv)[:, np.newaxis]
    y = random_range((NUM_AREA_POINTS,), -maxv, -minv)[:, np.newaxis]
    bottom_left = np.hstack([x,y])

    datas = np.vstack([top_left, top_right, bottom_right, bottom_left])
    np.random.shuffle(datas)

    return datas




def show(org_data, num_clusters, cls, centers):
    fig, ax_lst = plt.subplots(1, 2)
    org_ax, cls_ax = ax_lst

    org_ax.plot(org_data[:, 0], org_data[:, 1], 'o')
    for i in range(num_clusters):
        cls_data = org_data[cls == i]
        if len(cls_data) == 0: continue
        cls_ax.plot(cls_data[:, 0], cls_data[:, 1], linestyle='None', marker='o')

    cls_ax.plot(centers[:, 0], centers[:, 1], linestyle='None', marker='v', color='k', markersize=10)

    plt.show()


NUM_CLUSTERS = 4
MAX_ITERS = 10

org_data = gen_datas(10)
cls, centers = kmean(org_data, NUM_CLUSTERS, MAX_ITERS)
show(org_data, NUM_CLUSTERS, cls, centers)