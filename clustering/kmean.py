
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

        yield cls, np.squeeze(c, axis=0)


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
            num_points = np.random.randint(*area_point_range)
            x = random_range((num_points,), min_x, max_x)[:, np.newaxis]
            y = random_range((num_points,), min_y, max_y)[:, np.newaxis]
            datas.append(np.hstack([x,y]))
            labels.append(np.ones(num_points) * (i * num_area[1] + j))

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


MAP_BOUNDS = (-10, -10, 10, 10)
NUM_AREA = (2, 2)
AREA_POINTS_RANGE = (5, 10)
MAX_ITERS = 10
SHOW_PROGRESS = True
NUM_CLUSTERS = NUM_AREA[0] * NUM_AREA[1]

datas, labels = gen_datas(MAP_BOUNDS, NUM_AREA, AREA_POINTS_RANGE)
f_kmean = kmean(datas, NUM_CLUSTERS, MAX_ITERS)

iters = 0
def press_callback(event):
    global f_kmean, iters
    cls = centers = None
    try:
        while True:
            cls, centers = next(f_kmean)
            iters += 1
            if SHOW_PROGRESS: break

    except StopIteration:
        pass

    if cls is not None: show(MAP_BOUNDS, datas, NUM_CLUSTERS, cls, centers, iters, press_callback)

show(MAP_BOUNDS, datas, NUM_CLUSTERS, None, None, 0, press_callback)