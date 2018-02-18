import numpy as np
import random
import matplotlib.pyplot as plt


def lvq(datas, labels, learning_rate, max_iters, batch_iters=None):
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

    # train
    for i in range(max_iters):
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


    return p_x, p_y



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


def show(map_bounds, datas, classes, proto_vectors, iters, click_callback):
    global figure, axes
    try:
        axes.cla()
    except:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    axes.axis([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]])
    axes.set_title("Iterators: {}".format(iters))

    # register press event
    cid = figure.canvas.mpl_connect('button_press_event', click_callback)

    # draw category datas
    if classes is not None:
        for c in np.unique(classes):
            cls_data = datas[classes == c]
            if len(cls_data) == 0: continue
            axes.plot(cls_data[:, 0], cls_data[:, 1], linestyle='None', marker='o')

    # draw proto vectors
    axes.plot(proto_vectors[:, 0], proto_vectors[:, 1], linestyle='None', marker='+', color='k', markersize=10)



    plt.show()


MAP_BOUNDS = (-10, -10, 10, 10)
NUM_AREA = (2, 2)
AREA_POINTS_RANGE = (5, 10)
MAX_ITERS = 10
SHOW_PROGRESS = True
NUM_CLUSTERS = NUM_AREA[0] * NUM_AREA[1]

datas, labels = gen_datas(MAP_BOUNDS, NUM_AREA, AREA_POINTS_RANGE)
pv_x, pv_y = lvq(datas, labels, 0.3, MAX_ITERS, batch_iters=None)


iters = 0
def press_callback(event):
    if event.xdata is None: return
    global datas, labels
    # inference
    v = np.array([event.xdata, event.ydata])
    d = np.sqrt(np.sum(np.square(v - pv_x), axis=1))
    min_index = np.argmin(d)
    cls = pv_y[min_index]

    # update datas
    datas = np.vstack([datas, v[np.newaxis, :]])
    labels = np.hstack([labels, cls])
    show(MAP_BOUNDS, datas, labels, pv_x, MAX_ITERS, press_callback)


show(MAP_BOUNDS, datas, labels, pv_x, MAX_ITERS, press_callback)