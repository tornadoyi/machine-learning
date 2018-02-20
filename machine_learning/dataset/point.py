import numpy as np

def _random_range(shape, min, max):
    assert max > min
    x = np.random.rand(*shape)
    return min + (max - min) * x


def _shuffle_datas(datas, labels):
    indexes = np.arange(len(datas))
    np.random.shuffle(indexes)
    return datas[indexes], labels[indexes]



def area(bounds, num_area, num_area_points, area_border_ratio=0):
    """
        Generate points in mutiple rectangle areas, label is area index.
    :param bounds: tuple
                Area will be divided by bounds with (min_x, min_y, width, height)
    :param num_area: tuple
                The count of area (count in x-axis, count in y-axis)
    :param num_area_points: int/tuple/list
                Specific number with integer or a range with tuple or list
    :param area_border_ratio:
                The ratio of border
    :return: ndarray, ndarry
                Datas and labels
    """
    if isinstance(num_area_points, (tuple, list)):
        num_area_points = list(num_area_points)
    elif isinstance(num_area_points, int):
        num_area_points = [num_area_points, num_area_points + 1]
    else: raise Exception('Type of num_area_points should be int, list or tuple')

    area_width = bounds[2] /  num_area[0]
    area_height = bounds[3] / num_area[1]

    datas = []
    labels = []
    for i in range(num_area[0]):
        for j in range(num_area[1]):
            min_x, min_y = bounds[0] + area_width * i, bounds[1] + area_height * j
            max_x, max_y = min_x + area_width, min_y + area_height
            min_x += area_width * area_border_ratio
            max_x -= area_width * area_border_ratio
            min_y += area_height * area_border_ratio
            max_y -= area_height * area_border_ratio
            num_points = np.random.randint(*num_area_points)
            x = _random_range((num_points,), min_x, max_x)[:, np.newaxis]
            y = _random_range((num_points,), min_y, max_y)[:, np.newaxis]
            datas.append(np.hstack([x,y]))
            labels.append(np.ones(num_points, dtype=np.int) * (i * num_area[1] + j))

    return _shuffle_datas(np.vstack(datas), np.hstack(labels))


def circle(centers, radius, counts):
    """
        Generate circle points, label is circle index
    :param centers: tuple/list
                Center list with (x, y) element
    :param radius:
                Radius of circle
    :param counts:
                Number of points will be generate in each circle
    :return: ndarray, ndarry
                Datas and labels
    """
    circle_count = np.max([len(centers), len(radius), len(counts)])
    assert len(centers) == 1 or len(centers) == circle_count
    assert len(radius) == 1 or len(radius) == circle_count
    assert len(counts) == 1 or len(counts) == circle_count

    if len(centers) != circle_count: centers = list(centers) * circle_count
    if len(radius) != circle_count: radius = list(radius) * circle_count
    if len(counts) != circle_count: counts = list(counts) * circle_count

    datas, labels = [], []
    for i in range(circle_count):
        o = centers[i]
        r = radius[i]
        c = counts[i]
        p = np.array([r, 0])
        count_range = c if isinstance(c, (tuple, list)) else (c, c+1)
        num_points = np.random.randint(*count_range)

        theta = _random_range((num_points, ), 0, 360)
        cos, sin = np.cos(theta)[:, np.newaxis], np.sin(theta)[:, np.newaxis]
        transform = np.hstack([cos, -sin, sin, cos]).reshape([num_points, 2, 2])
        v = np.tensordot(transform, p[np.newaxis, :, np.newaxis], axes=[[2], [1]])
        p1 = o + np.squeeze(np.squeeze(v, axis=2), axis=2)
        datas.append(p1)
        labels.append(np.array([i] * c))

    return _shuffle_datas(np.vstack(datas), np.hstack(labels))
